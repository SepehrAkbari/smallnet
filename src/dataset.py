import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CamVidDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_dict_path, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg'))])
        
        # Build the color mapping dictionary from the CSV
        self.color_to_index = self._load_color_map(class_dict_path)

    def _load_color_map(self, path):
        color_to_idx = {}
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip the header row
            for idx, row in enumerate(reader):
                # CSV format: name, r, g, b
                r, g, b = int(row[1]), int(row[2]), int(row[3])
                color_to_idx[(r, g, b)] = idx
        return color_to_idx

    def _rgb_to_index(self, mask_np):
        # Create an empty 2D array for the class indices
        index_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
        
        # Vectorized mapping: find all pixels matching a color and assign the class index
        for rgb, idx in self.color_to_index.items():
            matches = (mask_np == rgb).all(axis=-1)
            index_mask[matches] = idx
            
        return index_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB") # Force RGB to match the CSV
        
        if self.transform:
            image = self.transform(image)
        
        # Convert RGB mask to Index mask
        mask_np = np.array(mask)
        index_mask = self._rgb_to_index(mask_np)
        mask_tensor = torch.from_numpy(index_mask).long()
        
        # Resize mask to 352x480 to match network output
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(), 
            size=(352, 480), 
            mode='nearest'
        ).squeeze().long()

        # Any artifact pixels that didn't perfectly match a class color stay 0 (usually background/void)
        # So we no longer need the extreme 255 clamping here.
        return image, mask_tensor