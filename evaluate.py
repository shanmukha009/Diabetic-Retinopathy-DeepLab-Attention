#!/usr/bin/env python3
"""
FIXED evaluation script for DR lesion segmentation models.
Now supports UNet, DeepLab, Attention U-Net, and PSPNet architectures.
"""

# All imports are at the top level
import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports
import config
from models.deeplab import get_deeplab_model
from models.attention_unet import get_attention_unet
from models.unet import get_unet_model
from models.pspnet import get_pspnet_model
from data.dataloader import get_dataloader
from utils.metrics import MetricTracker
from utils.visualization import visualize_predictions, save_visualization

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DR lesion segmentation models')
    parser.add_argument('--model', type=str, default='deeplab', 
                        choices=['deeplab', 'attention_unet', 'unet', 'pspnet'],
                        help='Model architecture to use')
    parser.add_argument('--encoder', type=str, default='resnet50', 
                        help='Encoder backbone (resnet34, resnet50) - for DeepLab and PSPNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for evaluation')
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT,
                        help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default=os.path.join(config.OUTPUT_DIR, 'evaluation'),
                        help='Directory to save evaluation results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save visualization of predictions')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU ID(s) to use, comma-separated')
    
    # UNet specific arguments (must match training configuration)
    parser.add_argument('--unet_base_channels', type=int, default=64,
                        help='Base number of channels for UNet (must match training)')
    parser.add_argument('--unet_bilinear', action='store_true', default=True,
                        help='Use bilinear upsampling in UNet (must match training)')
    parser.add_argument('--unet_dropout', action='store_true',
                        help='Use dropout in UNet (must match training)')
    parser.add_argument('--unet_dropout_rate', type=float, default=0.1,
                        help='Dropout rate for UNet (must match training)')
    
    # PSPNet specific arguments (must match training configuration)
    parser.add_argument('--pspnet_aux_loss', action='store_true', default=True,
                        help='Use auxiliary loss in PSPNet (must match training)')
    parser.add_argument('--pspnet_lesion_specific', action='store_true',
                        help='Use lesion-specific PSPNet variant (must match training)')
    
    return parser.parse_args()

def debug_model_loading(model, checkpoint_path):
    """Debug model loading to see what weights are missing/unexpected"""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Get model's current state dict
    current_state = model.state_dict()
    
    # Find missing and unexpected keys
    missing_keys = []
    unexpected_keys = []
    size_mismatches = []
    
    for key in model_state.keys():
        if key not in current_state:
            unexpected_keys.append(key)
        elif model_state[key].shape != current_state[key].shape:
            size_mismatches.append(f"{key}: checkpoint {model_state[key].shape} vs model {current_state[key].shape}")
    
    for key in current_state.keys():
        if key not in model_state:
            missing_keys.append(key)
    
    print("=== MODEL LOADING DEBUG ===")
    print(f"Missing keys in checkpoint: {len(missing_keys)}")
    for key in missing_keys[:10]:  # Show first 10
        print(f"  - {key}")
    
    print(f"\nUnexpected keys in checkpoint: {len(unexpected_keys)}")
    for key in unexpected_keys[:10]:  # Show first 10
        print(f"  - {key}")
    
    print(f"\nSize mismatches: {len(size_mismatches)}")
    for mismatch in size_mismatches[:10]:  # Show first 10
        print(f"  - {mismatch}")
    
    return missing_keys, unexpected_keys, size_mismatches

def evaluate(model, dataloader, device, output_dir, save_visualizations=False):
    """
    Evaluate the model on a dataset.
    """
    model.eval()
    metric_tracker = MetricTracker(config.LESION_TYPES)
    
    # Create visualization directory if needed
    vis_dir = os.path.join(output_dir, 'visualizations')
    if save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            img_ids = batch['img_id']
            
            if batch_idx == 0:
                print(f"First item - image shape: {images[0].shape}, mask shape: {masks[0].shape}")
            
            # Forward pass - handle different model outputs
            model_output = model(images)
            
            # Handle models with deep supervision
            if isinstance(model_output, tuple):
                outputs = model_output[0]  # Main output
            else:
                outputs = model_output
            
            # Apply sigmoid
            outputs_sigmoid = torch.sigmoid(outputs)
            
            # Track metrics
            metric_tracker.update(outputs_sigmoid, masks)
            
            # Save visualizations if requested
            if save_visualizations and batch_idx < 10:  # Only save first 10 batches to avoid too many images
                for i in range(len(img_ids)):
                    img = images[i].detach().cpu()
                    mask = masks[i].detach().cpu()
                    pred = outputs_sigmoid[i].detach().cpu()
                    
                    # Create visualization
                    vis = visualize_predictions(
                        img.unsqueeze(0), 
                        mask.unsqueeze(0), 
                        pred.unsqueeze(0), 
                        config.LESION_TYPES
                    )
                    
                    # Save visualization
                    save_path = os.path.join(vis_dir, f"{img_ids[i]}.png")
                    save_visualization(vis, save_path)
    
    # Get metrics
    metrics = metric_tracker.get_metrics()
    
    # Calculate per-lesion metrics
    lesion_metrics = {}
    for lesion in config.LESION_TYPES:
        lesion_metrics[lesion] = {
            'AP': metrics[f'ap_{lesion}'],
            'IoU': metrics[f'iou_{lesion}'],
            'Dice': metrics[f'dice_{lesion}']
        }
    
    # Return metrics
    return {
        'mean_ap': metrics['mean_ap'],
        'mean_iou': metrics['mean_iou'],
        'mean_dice': metrics['mean_dice'],
        'lesion_metrics': lesion_metrics
    }

def load_model_from_checkpoint(args, device):
    """
    Load model from checkpoint with proper architecture reconstruction.
    
    Args:
        args: Command line arguments
        device: Device to load model on
        
    Returns:
        Loaded model and checkpoint
    """
    num_classes = len(config.LESION_TYPES)
    
    # Create model based on architecture
    if args.model == 'deeplab':
        model = get_deeplab_model(
            num_classes=num_classes,
            backbone=args.encoder,
            pretrained=False
        )
        print(f"Created DeepLab model with {args.encoder} backbone")
        
    elif args.model == 'attention_unet':
        # Attention U-Net is hardcoded to ResNet50
        if args.encoder != 'resnet50':
            print(f"WARNING: attention_unet.py is hardcoded to ResNet50, ignoring --encoder {args.encoder}")
        
        model = get_attention_unet(
            num_classes=num_classes,
            pretrained_encoder=False  # Don't need pretrained when loading checkpoint
        )
        print("Created Attention U-Net model")
        
    elif args.model == 'unet':
        model = get_unet_model(
            num_classes=num_classes,
            in_channels=3,
            bilinear=args.unet_bilinear,
            base_channels=args.unet_base_channels,
            with_dropout=args.unet_dropout,
            dropout_rate=args.unet_dropout_rate
        )
        print(f"Created U-Net model:")
        print(f"  - Base channels: {args.unet_base_channels}")
        print(f"  - Bilinear upsampling: {args.unet_bilinear}")
        print(f"  - Dropout: {args.unet_dropout}")
        if args.unet_dropout:
            print(f"  - Dropout rate: {args.unet_dropout_rate}")
    
    elif args.model == 'pspnet':
        model = get_pspnet_model(
            num_classes=num_classes,
            backbone=args.encoder,
            pretrained=False,  # Don't need pretrained when loading checkpoint
            aux_loss=args.pspnet_aux_loss,
            lesion_specific=args.pspnet_lesion_specific,
            lesion_types=config.LESION_TYPES if args.pspnet_lesion_specific else None
        )
        print(f"Created PSPNet model:")
        print(f"  - Backbone: {args.encoder}")
        print(f"  - Auxiliary loss: {args.pspnet_aux_loss}")
        print(f"  - Lesion-specific: {args.pspnet_lesion_specific}")
            
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Check model type
    print(f"Model created, type: {type(model)}")
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module instance, got {type(model)}")
    
    # Load checkpoint with multiple fallback methods
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = None
    try:
        # Direct loading approach for backwards compatibility
        checkpoint = torch.load(args.checkpoint, map_location=device)
        print("Loaded checkpoint using default parameters")
    except Exception as e:
        print(f"Error with default loading: {e}")
        try:
            # Try explicitly with weights_only=False
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            print("Loaded checkpoint with weights_only=False")
        except Exception as e2:
            print(f"Error with weights_only=False: {e2}")
            try:
                # Try with pickle globals allowlisted
                from torch import serialization
                with serialization.safe_globals(["numpy._core.multiarray.scalar"]):
                    checkpoint = torch.load(args.checkpoint, map_location=device)
                print("Loaded checkpoint with safe_globals")
            except Exception as e3:
                print(f"All loading methods failed. Final error: {e3}")
                raise RuntimeError("Failed to load checkpoint with all methods")
    
    if checkpoint is None:
        raise RuntimeError("Failed to load checkpoint - checkpoint is None")
    
    # Print checkpoint contents
    print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dictionary'}")
    
    # Check if checkpoint contains training parameters for model-specific configurations
    if isinstance(checkpoint, dict) and 'train_params' in checkpoint:
        train_params = checkpoint['train_params']
        print("Found training parameters in checkpoint:")
        
        # Verify parameters match for different models
        param_mismatches = []
        
        if args.model == 'unet':
            if 'unet_base_channels' in train_params:
                if train_params['unet_base_channels'] != args.unet_base_channels:
                    param_mismatches.append(f"base_channels: checkpoint={train_params['unet_base_channels']}, specified={args.unet_base_channels}")
            
            if 'unet_bilinear' in train_params:
                if train_params['unet_bilinear'] != args.unet_bilinear:
                    param_mismatches.append(f"bilinear: checkpoint={train_params['unet_bilinear']}, specified={args.unet_bilinear}")
            
            if 'unet_dropout' in train_params:
                if train_params['unet_dropout'] != args.unet_dropout:
                    param_mismatches.append(f"dropout: checkpoint={train_params['unet_dropout']}, specified={args.unet_dropout}")
        
        elif args.model == 'pspnet':
            if 'pspnet_aux_loss' in train_params:
                if train_params['pspnet_aux_loss'] != args.pspnet_aux_loss:
                    param_mismatches.append(f"aux_loss: checkpoint={train_params['pspnet_aux_loss']}, specified={args.pspnet_aux_loss}")
            
            if 'pspnet_lesion_specific' in train_params:
                if train_params['pspnet_lesion_specific'] != args.pspnet_lesion_specific:
                    param_mismatches.append(f"lesion_specific: checkpoint={train_params['pspnet_lesion_specific']}, specified={args.pspnet_lesion_specific}")
        
        if param_mismatches:
            print("WARNING: Parameter mismatches detected!")
            for mismatch in param_mismatches:
                print(f"  - {mismatch}")
            print("Model loading might fail. Use the same parameters as training.")
    
    # Debug model loading
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        missing, unexpected, mismatches = debug_model_loading(model, args.checkpoint)
        
        if mismatches:
            print("\n⚠️  WARNING: Architecture mismatch detected!")
            print("The checkpoint was likely trained with different parameters.")
            if args.model == 'deeplab':
                print(f"Current model uses encoder: {args.encoder}")
            elif args.model == 'unet':
                print(f"Current UNet config: channels={args.unet_base_channels}, bilinear={args.unet_bilinear}, dropout={args.unet_dropout}")
            elif args.model == 'pspnet':
                print(f"Current PSPNet config: encoder={args.encoder}, aux_loss={args.pspnet_aux_loss}, lesion_specific={args.pspnet_lesion_specific}")
            print("Consider using the correct parameters or retraining.")
    
    # Load model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        try:
            # Try loading the state dict - some keys may not match exactly
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Successfully loaded model_state_dict with strict=False")
        except Exception as e:
            print(f"Error loading model_state_dict: {e}")
            try:
                # Try partial loading
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                
                # Debug: print key differences
                print("Checkpoint keys:", len(pretrained_dict.keys()))
                print("Model keys:", len(model_dict.keys()))
                
                # Filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                print(f"Matched keys: {len(pretrained_dict.keys())}")
                
                # Update model state dict
                model_dict.update(pretrained_dict) 
                model.load_state_dict(model_dict)
                print("Successfully loaded model_state_dict with partial loading")
            except Exception as e2:
                print(f"All state_dict loading methods failed: {e2}")
                raise
    else:
        try:
            # Try direct loading if state_dict is the full checkpoint
            model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded checkpoint directly as state_dict")
        except Exception as e:
            print(f"Error loading checkpoint directly: {e}")
            raise RuntimeError("Failed to load model weights")
    
    return model, checkpoint

def run_evaluation():
    """
    Main evaluation function.
    """
    args = parse_args()
    
    # Set CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Evaluating model: {args.model}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataloader
    print(f"Loading {args.split} dataset from {args.data_root}")
    dataloader = get_dataloader(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS
    )
    
    # Load model
    print(f"Creating {args.model} model...")
    model, checkpoint = load_model_from_checkpoint(args, device)
    
    # Move model to device
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Print checkpoint info if available
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch'] + 1}")
        if 'val_ap' in checkpoint:
            print(f"Checkpoint validation AP: {checkpoint['val_ap']:.4f}")
        elif 'val_metrics' in checkpoint and 'mean_ap' in checkpoint['val_metrics']:
            print(f"Checkpoint validation AP: {checkpoint['val_metrics']['mean_ap']:.4f}")
    
    # Evaluate model
    print(f"Evaluating {args.model} on {args.split} set...")
    metrics = evaluate(
        model, 
        dataloader, 
        device, 
        output_dir, 
        save_visualizations=args.save_visualizations
    )
    
    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset split: {args.split}")
    print(f"Mean AP: {metrics['mean_ap']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")
    
    print("\nLesion-specific metrics:")
    for lesion, lesion_metrics in metrics['lesion_metrics'].items():
        print(f"{lesion}:")
        print(f"  AP: {lesion_metrics['AP']:.4f}")
        print(f"  IoU: {lesion_metrics['IoU']:.4f}")
        print(f"  Dice: {lesion_metrics['Dice']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{args.model}_{args.split}_metrics.json")
    
    # Add evaluation metadata
    evaluation_info = {
        'model': args.model,
        'split': args.split,
        'checkpoint_path': args.checkpoint,
        'evaluation_params': vars(args),
        'metrics': metrics
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_info, f, indent=4, cls=NumpyEncoder)
    
    print(f"\nResults saved to: {metrics_path}")
    if args.save_visualizations:
        vis_dir = os.path.join(output_dir, 'visualizations')
        print(f"Visualizations saved to: {vis_dir}")

class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Main execution
if __name__ == "__main__":
    run_evaluation()