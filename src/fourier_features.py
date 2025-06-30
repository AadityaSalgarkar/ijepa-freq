
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class FourierFeatures(nn.Module):
    def __init__(self, n_fourier=8):
        """
        Initializes the FourierFeatures module.

        Args:
            n_fourier (int): The number of Fourier feature channels to add.
        """
        super().__init__()
        self.n_fourier = n_fourier

    def apply_mask_and_ifft(self, f_shifted, mask):
        # Apply the masks to the Fourier-transformed image.
        masked_f = f_shifted * mask.to(f_shifted.device)

        # Convert the masked Fourier features back to the spatial domain.
        f_ishifted = torch.fft.ifftshift(masked_f)
        img_back = torch.fft.ifft2(f_ishifted)
        return img_back.real

    def __call__(self, x):
        """
        Adds Fourier features to the input image.

        Args:
            x (torch.Tensor): The input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: The image tensor with added Fourier feature channels.
        """
        # Assuming x is (C, H, W)

        # 1. Compute the 2D Fourier transform of the single channel image.
        f = torch.fft.fft2(x) # Shape (C, H, W)
        f_shifted = torch.fft.fftshift(f) # Shape (C, H, W)

        _, h, w = x.shape # Use h, w from single channel
        center_h, center_w = h // 2, w // 2

        # 2. Create `n_fourier` masks for different frequency bands.
        fourier_features = []
        radii = [(i + 1) * min(center_h, center_w) / (self.n_fourier) for i in range(self.n_fourier - 1)]
        y, x_grid = np.ogrid[-center_h:h-center_h, -center_w:w-center_w]
        distance_from_center = torch.from_numpy(np.sqrt(x_grid**2 + y**2)).to(x.device) # Ensure on same device

        # First filter: inner circle
        mask = torch.zeros((1, h, w), device=x.device) # Mask should be (1, H, W)
        mask_area = distance_from_center <= radii[0]
        mask[0, mask_area] = 1 # Apply to the single channel
        fourier_features.append(self.apply_mask_and_ifft(f_shifted, mask)) # This will return (1, H, W)

        # n-2 filters: discs between consecutive circles
        for i in range(self.n_fourier - 2):
            mask = torch.zeros((1, h, w), device=x.device) # Mask should be (1, H, W)
            inner_radius = radii[i]
            outer_radius = radii[i+1]
            mask_area = (distance_from_center > inner_radius) & (distance_from_center <= outer_radius)
            mask[0, mask_area] = 1 # Apply to the single channel
            fourier_features.append(self.apply_mask_and_ifft(f_shifted, mask)) # This will return (1, H, W)

        # Last filter: exterior of the biggest circle
        mask = torch.ones((1, h, w), device=x.device) # Mask should be (1, H, W)
        mask_area = distance_from_center <= radii[-1]
        mask[0, mask_area] = 0 # Apply to the single channel
        fourier_features.append(self.apply_mask_and_ifft(f_shifted, mask)) # This will return (1, H, W)

        fourier_features = torch.cat(fourier_features, dim=0)

        # Concatenate the fourier features with the original image
        x = torch.cat([x, fourier_features], dim=0)
        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(n_fourier={self.n_fourier})'

if __name__ == '__main__':
    # Load the image
    img = Image.open('/Users/aaditya/repos/ijepa/images/cat.jpg')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)

    # Add Fourier features
    fourier_features_transform = FourierFeatures(n_fourier=3)
    img_with_features = fourier_features_transform(img_tensor)

    # Save the original image and all feature channels
    num_channels = img_with_features.shape[0]
    for i in range(num_channels):
        channel_img = transforms.ToPILImage()(img_with_features[i].unsqueeze(0))
        channel_img.save(f'/Users/aaditya/repos/ijepa/images/cat_{i}.jpg')
