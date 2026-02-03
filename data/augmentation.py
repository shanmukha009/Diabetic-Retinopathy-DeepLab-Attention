"""
Data augmentation strategies for improving segmentation performance.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

def get_train_augmentation():
    """
    Returns the augmentation pipeline for training.
    
    Includes various transformations to improve model robustness,
    especially for small lesions like microaneurysms (MA).
    """
    return A.Compose([
        A.Resize(height=config.IMG_SIZE[0], width=config.IMG_SIZE[1]),
        
        # Spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, 
            scale_limit=0.1, 
            rotate_limit=15, 
            p=0.5
        ),
        
        # Color augmentations - important for dealing with different fundus image qualities
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.8),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=15, 
                val_shift_limit=10, 
                p=0.8
            ),
        ], p=0.5),
        
        # Noise augmentations - helps with robustness to image quality variations
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        
        # Special augmentations for small lesions like microaneurysms
        A.OneOf([
            A.ElasticTransform(
                alpha=1, 
                sigma=50, 
                alpha_affine=50, 
                p=0.5
            ),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(
                distort_limit=1.0,
                shift_limit=0.5,
                p=0.5
            ),
        ], p=0.3),
        
        # Normalization and conversion to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_valid_augmentation():
    """
    Returns the augmentation pipeline for validation/testing.
    Only includes resizing and normalization.
    """
    return A.Compose([
        A.Resize(height=config.IMG_SIZE[0], width=config.IMG_SIZE[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])