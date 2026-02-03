"""
Training script for DR lesion segmentation models.
"""

import os
import time
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local imports
import config
from models.deeplab import get_deeplab_model
from models.attention_unet import get_attention_unet
from models.unet import get_unet_model
from data.dataloader import get_dataloader
from utils.losses import CombinedLoss
from utils.metrics import MetricTracker
from utils.visualization import visualize_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Train DR lesion segmentation models')
    parser.add_argument('--model', type=str, default='deeplab', 
                        choices=['deeplab', 'attention_unet', 'unet'],
                        help='Model architecture to use')
    parser.add_argument('--encoder', type=str, default='resnet50', 
                        help='Encoder backbone (for DeepLab)')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT,
                        help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help='Directory to save results')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU ID(s) to use, comma-separated')
    
    # UNet specific arguments
    parser.add_argument('--unet_base_channels', type=int, default=64,
                        help='Base number of channels for UNet (default: 64)')
    parser.add_argument('--unet_bilinear', action='store_true', default=True,
                        help='Use bilinear upsampling in UNet (default: True)')
    parser.add_argument('--unet_dropout', action='store_true',
                        help='Use dropout in UNet')
    parser.add_argument('--unet_dropout_rate', type=float, default=0.1,
                        help='Dropout rate for UNet if --unet_dropout is used')
    
    return parser.parse_args()

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, writer, metric_tracker):
    """
    Train the model for one epoch.
    """
    model.train()
    metric_tracker.reset()
    
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass - handle different model outputs
        model_name = model.__class__.__name__
        
        if model_name in ['DeepLabV3Plus', 'AttentionUNet', 'UNet']:
            # Models with deep supervision
            model_output = model(images)
            
            if isinstance(model_output, tuple):
                outputs, aux_outputs = model_output
                
                # Compute loss for main output
                loss = loss_fn(outputs, masks)
                
                # Add auxiliary losses if deep supervision is used
                if isinstance(aux_outputs, tuple):
                    for aux_output in aux_outputs:
                        loss += 0.4 * loss_fn(aux_output, masks)
                else:
                    loss += 0.4 * loss_fn(aux_outputs, masks)
            else:
                # Single output
                outputs = model_output
                loss = loss_fn(outputs, masks)
        else:
            # Standard models with single output
            outputs = model(images)
            loss = loss_fn(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            outputs_sigmoid = torch.sigmoid(outputs)
            metric_tracker.update(outputs_sigmoid, masks)
        
        # Update progress bar
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/(batch_idx+1))
    
    # Compute average loss and metrics
    epoch_loss = running_loss / len(dataloader)
    metrics = metric_tracker.get_metrics()
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f'Metrics/train/{metric_name}', metric_value, epoch)
    
    # Log sample predictions
    if (epoch + 1) % 5 == 0:
        images_sample = images[:4].detach().cpu()
        masks_sample = masks[:4].detach().cpu()
        outputs_sample = outputs_sigmoid[:4].detach().cpu()
        
        visualization = visualize_predictions(
            images_sample, masks_sample, outputs_sample, config.LESION_TYPES
        )
        writer.add_image('Predictions/train', visualization, epoch, dataformats='HWC')
    
    return epoch_loss, metrics

def validate(model, dataloader, loss_fn, device, epoch, writer, metric_tracker):
    """
    Validate the model on the validation set.
    """
    model.eval()
    metric_tracker.reset()
    
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Valid]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass - handle different model outputs
            model_name = model.__class__.__name__
            
            if model_name in ['DeepLabV3Plus', 'AttentionUNet', 'UNet']:
                # Models that may have deep supervision
                model_output = model(images)
                
                if isinstance(model_output, tuple):
                    # Deep supervision outputs during training
                    outputs, _ = model_output
                else:
                    # Single output
                    outputs = model_output
            else:
                # Standard models
                outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, masks)
            
            # Track metrics
            outputs_sigmoid = torch.sigmoid(outputs)
            metric_tracker.update(outputs_sigmoid, masks)
            
            # Update progress bar
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/(batch_idx+1))
    
    # Compute average loss and metrics
    epoch_loss = running_loss / len(dataloader)
    metrics = metric_tracker.get_metrics()
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f'Metrics/val/{metric_name}', metric_value, epoch)
    
    # Log sample predictions
    if (epoch + 1) % 5 == 0:
        images_sample = images[:4].detach().cpu()
        masks_sample = masks[:4].detach().cpu()
        outputs_sample = outputs_sigmoid[:4].detach().cpu()
        
        visualization = visualize_predictions(
            images_sample, masks_sample, outputs_sample, config.LESION_TYPES
        )
        writer.add_image('Predictions/val', visualization, epoch, dataformats='HWC')
    
    return epoch_loss, metrics

def main():
    """
    Main training function.
    """
    args = parse_args()
    
    # Set CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training model: {args.model}")
    
    # Create output directories
    output_dir = args.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Save training parameters
    with open(os.path.join(output_dir, 'train_params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        split='train',
        batch_size=args.batch_size
    )
    
    val_loader = get_dataloader(
        data_root=args.data_root,
        split='valid',
        batch_size=args.batch_size
    )
    
    # Create model
    num_classes = len(config.LESION_TYPES)
    
    if args.model == 'deeplab':
        model = get_deeplab_model(
            num_classes=num_classes,
            backbone=args.encoder,
            pretrained=True
        )
        print(f"Created DeepLab model with {args.encoder} backbone")
        
    elif args.model == 'attention_unet':
        model = get_attention_unet(
            num_classes=num_classes,
            pretrained_encoder=True
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
        
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Move model to device
    model = model.to(device)
    
    # Create loss function
    loss_fn = CombinedLoss(
        lesion_types=config.LESION_TYPES,
        weights={
            'dice': 1.0,
            'bce': 0.5,
            'focal': 1.0,
            'boundary': 0.5
        }
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE
    )
    
    # Create metric tracker
    metric_tracker = MetricTracker(config.LESION_TYPES)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_ap = 0.0
    best_val_metrics = None
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs")
    print("-" * 50)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, writer, metric_tracker
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, loss_fn, device, epoch, writer, metric_tracker
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Mean AP: {train_metrics['mean_ap']:.4f}")
        print(f"Valid Loss: {val_loss:.4f}, Mean AP: {val_metrics['mean_ap']:.4f}")
        print(f"Individual APs - MA: {val_metrics['ap_MA']:.4f}, EX: {val_metrics['ap_EX']:.4f}, SE: {val_metrics['ap_SE']:.4f}, HE: {val_metrics['ap_HE']:.4f}")
        
        # Save model based on validation AP (primary) or loss (secondary)
        current_val_ap = val_metrics['mean_ap']
        is_best = False
        
        # Primary criterion: AP improvement
        if current_val_ap > best_val_ap:
            best_val_ap = current_val_ap
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            is_best = True
            patience_counter = 0
        # Secondary criterion: loss improvement (if AP is similar)
        elif abs(current_val_ap - best_val_ap) < 0.01 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            is_best = True
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save best model
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'val_ap': current_val_ap,
                'train_params': vars(args)
            }, checkpoint_path)
            
            print(f"✓ Saved best model (AP: {current_val_ap:.4f}, Loss: {val_loss:.4f})")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'train_params': vars(args)
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation mean AP: {best_val_ap:.4f}")
    if best_val_metrics:
        print(f"Best validation mean IoU: {best_val_metrics['mean_iou']:.4f}")
        print(f"Best validation mean Dice: {best_val_metrics['mean_dice']:.4f}")
        print("\nBest individual lesion APs:")
        for lesion_type in config.LESION_TYPES:
            print(f"  {lesion_type}: {best_val_metrics[f'ap_{lesion_type}']:.4f}")
    
    # Close TensorBoard writer
    writer.close()
    print(f"\nLogs saved to: {log_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

if __name__ == "__main__":
    main()