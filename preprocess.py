"""
Preprocessing script for the DDR dataset.
Performs tasks like:
1. Converting TIF masks to PNG for easier handling
2. Checking dataset integrity
3. Creating dataset splits if needed
"""

import os
import shutil
import argparse
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
import json
import glob

import config

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the DDR dataset')
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT,
                        help='Root directory of the DDR dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (if None, modify in-place)')
    parser.add_argument('--convert_tif', action='store_true',
                        help='Convert TIF masks to PNG format')
    parser.add_argument('--check_integrity', action='store_true',
                        help='Check dataset integrity')
    parser.add_argument('--create_splits', action='store_true',
                        help='Create train/val/test splits if not already present')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def convert_tif_to_png(data_root, output_dir=None):
    """
    Convert TIF masks to PNG format for easier handling.
    
    Args:
        data_root: Root directory of the DDR dataset
        output_dir: Output directory (if None, modify in-place)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all TIF files in the dataset
    tif_files = []
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_root, split, 'label')
        if not os.path.exists(split_dir):
            continue
            
        for lesion_type in config.LESION_TYPES:
            lesion_dir = os.path.join(split_dir, lesion_type)
            if not os.path.exists(lesion_dir):
                continue
                
            tif_files.extend(glob.glob(os.path.join(lesion_dir, '*.tif')))
    
    print(f"Found {len(tif_files)} TIF files to convert")
    
    # Convert each TIF file to PNG
    for tif_file in tqdm(tif_files, desc="Converting TIF to PNG"):
        # Load TIF image
        tif_img = Image.open(tif_file)
        
        # Determine output path
        if output_dir:
            # Maintain the same directory structure in the output directory
            rel_path = os.path.relpath(tif_file, data_root)
            out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        else:
            # Replace in-place
            out_path = os.path.splitext(tif_file)[0] + '.png'
        
        # Save as PNG
        tif_img.save(out_path)

def check_dataset_integrity(data_root):
    """
    Check that the dataset has the expected structure and all necessary files.
    
    Args:
        data_root: Root directory of the DDR dataset
    """
    integrity_issues = []
    
    # Check that the required splits exist
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            integrity_issues.append(f"Missing split directory: {split_dir}")
            continue
        
        # Check for image and label directories
        image_dir = os.path.join(split_dir, 'image')
        label_dir = os.path.join(split_dir, 'label')
        
        if not os.path.exists(image_dir):
            integrity_issues.append(f"Missing image directory: {image_dir}")
        
        if not os.path.exists(label_dir):
            integrity_issues.append(f"Missing label directory: {label_dir}")
            continue
        
        # Check for lesion directories
        for lesion_type in config.LESION_TYPES:
            lesion_dir = os.path.join(label_dir, lesion_type)
            if not os.path.exists(lesion_dir):
                integrity_issues.append(f"Missing lesion directory: {lesion_dir}")
        
        # Check that each image has corresponding mask files
        if os.path.exists(image_dir) and os.path.exists(label_dir):
            images = glob.glob(os.path.join(image_dir, '*.jpg')) + \
                     glob.glob(os.path.join(image_dir, '*.jpeg')) + \
                     glob.glob(os.path.join(image_dir, '*.png'))
            
            for image_path in tqdm(images, desc=f"Checking {split} images"):
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                
                for lesion_type in config.LESION_TYPES:
                    lesion_dir = os.path.join(label_dir, lesion_type)
                    if not os.path.exists(lesion_dir):
                        continue
                    
                    # Check if mask exists in any format
                    mask_found = False
                    for ext in ['.tif', '.png']:
                        mask_path = os.path.join(lesion_dir, image_name + ext)
                        if os.path.exists(mask_path):
                            mask_found = True
                            break
                    
                    if not mask_found:
                        integrity_issues.append(f"Missing mask for {image_name} in {lesion_type}")
    
    # Report integrity issues
    if integrity_issues:
        print(f"Found {len(integrity_issues)} integrity issues:")
        for issue in integrity_issues:
            print(f"  - {issue}")
    else:
        print("No integrity issues found.")
    
    return integrity_issues

def create_dataset_splits(data_root, output_dir=None, val_ratio=0.2, test_ratio=0.3, seed=42):
    """
    Create train/val/test splits if they don't already exist.
    
    Args:
        data_root: Root directory of the DDR dataset
        output_dir: Output directory (if None, create splits in place)
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
    """
    # Check if splits already exist
    splits_exist = True
    for split in ['train', 'valid', 'test']:
        if not os.path.exists(os.path.join(data_root, split)):
            splits_exist = False
            break
    
    if splits_exist:
        print("Dataset splits already exist.")
        return
    
    # Determine target directory
    target_dir = output_dir if output_dir else data_root
    
    # Find all image files
    image_dir = os.path.join(data_root, 'image')
    label_dir = os.path.join(data_root, 'label')
    
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print("ERROR: Cannot create splits. Expected 'image' and 'label' directories at the root level.")
        return
    
    # Get all image files
    images = glob.glob(os.path.join(image_dir, '*.jpg')) + \
             glob.glob(os.path.join(image_dir, '*.jpeg')) + \
             glob.glob(os.path.join(image_dir, '*.png'))
    
    image_names = [os.path.splitext(os.path.basename(img))[0] for img in images]
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(image_names)
    
    num_samples = len(image_names)
    num_test = int(num_samples * test_ratio)
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_test - num_val
    
    train_names = image_names[:num_train]
    val_names = image_names[num_train:num_train+num_val]
    test_names = image_names[num_train+num_val:]
    
    print(f"Creating splits: {num_train} train, {num_val} validation, {num_test} test")
    
    # Create split directories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(target_dir, split, 'image'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, 'label'), exist_ok=True)
        
        for lesion_type in config.LESION_TYPES:
            os.makedirs(os.path.join(target_dir, split, 'label', lesion_type), exist_ok=True)
    
    # Helper function to copy files for a split
    def copy_files_for_split(names, split):
        for name in tqdm(names, desc=f"Creating {split} split"):
            # Copy image
            for ext in ['.jpg', '.jpeg', '.png']:
                src_path = os.path.join(image_dir, name + ext)
                if os.path.exists(src_path):
                    dst_path = os.path.join(target_dir, split, 'image', name + ext)
                    shutil.copy2(src_path, dst_path)
                    break
            
            # Copy masks for each lesion type
            for lesion_type in config.LESION_TYPES:
                lesion_dir = os.path.join(label_dir, lesion_type)
                if not os.path.exists(lesion_dir):
                    continue
                
                for ext in ['.tif', '.png']:
                    src_path = os.path.join(lesion_dir, name + ext)
                    if os.path.exists(src_path):
                        dst_path = os.path.join(target_dir, split, 'label', lesion_type, name + ext)
                        shutil.copy2(src_path, dst_path)
                        break
    
    # Copy files for each split
    copy_files_for_split(train_names, 'train')
    copy_files_for_split(val_names, 'valid')
    copy_files_for_split(test_names, 'test')
    
    # Save split information
    split_info = {
        'train': train_names,
        'valid': val_names,
        'test': test_names,
        'split_params': {
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'seed': seed
        }
    }
    
    with open(os.path.join(target_dir, 'splits.json'), 'w') as f:
        json.dump(split_info, f, indent=4)
    
    print("Dataset splits created successfully.")

def main():
    """
    Main preprocessing function.
    """
    args = parse_args()
    
    if args.check_integrity:
        print("Checking dataset integrity...")
        check_dataset_integrity(args.data_root)
    
    if args.convert_tif:
        print("Converting TIF masks to PNG...")
        convert_tif_to_png(args.data_root, args.output_dir)
    
    if args.create_splits:
        print("Creating dataset splits...")
        create_dataset_splits(
            args.data_root, 
            args.output_dir,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()