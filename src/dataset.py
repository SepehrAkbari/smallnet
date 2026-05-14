import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CamVidDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        # Load mask. If it's a 3-channel RGB label, we convert it to 1-channel indices.
        # Many CamVid versions use 'P' mode (palette) or raw 8-bit labels.
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Ensure mask is (H, W) and LongTensor
        mask_np = np.array(mask)
        if len(mask_np.shape) == 3:
            # If your mask is RGB, you need a mapping. 
            # If it's already indexed but has a dummy 3rd dim, take the first channel.
            mask_np = mask_np[:, :, 0]
            
        mask_tensor = torch.from_numpy(mask_np).long()
        
        # Crop/Resize mask to match the 352x480 requirement
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(), 
            size=(352, 480), 
            mode='nearest'
        ).squeeze().long()

        return image, mask_tensor