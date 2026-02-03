"""
Dataloader for DDR lesion segmentation dataset.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .augmentation import get_train_augmentation, get_valid_augmentation
import config

class DDRLesionDataset(Dataset):
    """
    Dataset class for the DDR lesion segmentation dataset.
    """
    def __init__(self, data_root, split='train', lesion_types=None, transform=None):
        """
        Args:
            data_root: Root directory of the dataset
            split: 'train', 'valid', or 'test'
            lesion_types: List of lesion types to include (default all: MA, EX, SE, HE)
            transform: Albumentations transformations
        """
        if lesion_types is None:
            lesion_types = config.LESION_TYPES
            
        self.data_root = data_root
        self.split = split
        self.lesion_types = lesion_types
        self.transform = transform
        
        # Get the split directory
        self.split_dir = os.path.join(data_root, split)
        self.image_dir = os.path.join(self.split_dir, 'image')
        self.label_dir = os.path.join(self.split_dir, 'label')
        
        # Get all image filenames
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) 
                                       if f.endswith('.jpg')])
        
        print(f"Found {len(self.image_filenames)} images in {split} split")
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get image filename
        img_filename = self.image_filenames[idx]
        img_id = os.path.splitext(img_filename)[0]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize masks for each lesion type
        masks = {lesion_type: np.zeros(image.shape[:2], dtype=np.uint8) 
                 for lesion_type in self.lesion_types}
        
        # Load masks for each lesion type
        for lesion_type in self.lesion_types:
            lesion_dir = os.path.join(self.label_dir, lesion_type)
            # Look for the corresponding mask
            possible_mask_files = [
                f"{img_id}.tif",
                f"{img_id}.png"
            ]
            
            for mask_file in possible_mask_files:
                mask_path = os.path.join(lesion_dir, mask_file)
                if os.path.exists(mask_path):
                    if mask_path.endswith('.tif'):
                        # Use PIL for TIF files
                        mask = np.array(Image.open(mask_path))
                    else:
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Ensure mask is binary
                    if mask.max() > 1:
                        mask = (mask > 0).astype(np.uint8)
                    
                    masks[lesion_type] = mask
                    break
        
        # Create a multi-channel mask by stacking all lesion masks - stack in channel dimension
        # Change from (H, W, C) to (C, H, W) format expected by PyTorch
        multi_mask = np.stack([masks[lesion_type] for lesion_type in self.lesion_types], axis=0)
        
        # Apply transformations if available
        if self.transform:
            # For albumentations, we need to pass mask with (H, W, C)
            multi_mask_hwc = np.transpose(multi_mask, (1, 2, 0))
            transformed = self.transform(image=image, mask=multi_mask_hwc)
            image = transformed['image']
            
            # Convert back to (C, H, W) after transformation
            # Albumentations may return the mask in (H, W, C) format
            if transformed['mask'].ndim == 3 and transformed['mask'].shape[2] == len(self.lesion_types):
                # Already a tensor from Albumentations
                if torch.is_tensor(transformed['mask']):
                    multi_mask = transformed['mask'].permute(2, 0, 1)
                else:
                    # Still a numpy array
                    multi_mask = np.transpose(transformed['mask'], (2, 0, 1))
                    multi_mask = torch.from_numpy(multi_mask.astype(np.float32))
            else:
                # In case albumentations returns it differently
                multi_mask = transformed['mask']
                if not torch.is_tensor(multi_mask):
                    multi_mask = torch.from_numpy(multi_mask.astype(np.float32))
                
                # Ensure the mask has the right shape
                if multi_mask.ndim == 3 and multi_mask.shape[0] != len(self.lesion_types):
                    multi_mask = multi_mask.permute(2, 0, 1)
        else:
            # Basic conversion to tensor if no transformations
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
            multi_mask = torch.from_numpy(multi_mask.astype(np.float32))
        
        # Ensure mask is a tensor in (C, H, W) format
        if not torch.is_tensor(multi_mask):
            multi_mask = torch.from_numpy(multi_mask.astype(np.float32))
        
        # The issue is here - we need to check if it's already a tensor
        # FIX: Don't call astype on tensor objects
        
        # Debug information
        if idx == 0:
            print(f"First item - image shape: {image.shape}, mask shape: {multi_mask.shape}")
        
        return {
            'image': image,
            'mask': multi_mask,
            'img_id': img_id
        }

def get_dataloader(data_root=config.DATA_ROOT, split='train', batch_size=config.BATCH_SIZE, 
                   num_workers=config.NUM_WORKERS, lesion_types=None):
    """
    Get dataloader for a specific split.
    """
    if lesion_types is None:
        lesion_types = config.LESION_TYPES
    
    # Get appropriate augmentation
    if split == 'train':
        transform = get_train_augmentation()
    else:
        transform = get_valid_augmentation()
    
    # Create dataset
    dataset = DDRLesionDataset(
        data_root=data_root,
        split=split,
        lesion_types=lesion_types,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader