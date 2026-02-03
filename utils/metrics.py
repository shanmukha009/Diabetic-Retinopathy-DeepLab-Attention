"""
Evaluation metrics for lesion segmentation.
Based on the paper, we measure AP (Average Precision) and IoU (Intersection over Union).
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

def compute_iou(outputs, targets, smooth=1e-6, per_image=False):
    """
    Compute Intersection over Union (IoU) for each class.
    
    Args:
        outputs: Prediction tensor (N, C, H, W) with class probabilities
        targets: Ground truth tensor (N, C, H, W) with binary masks
        smooth: Smoothing factor to avoid division by zero
        per_image: If True, compute IoU for each image and then take the mean
        
    Returns:
        IoU for each class and mean IoU
    """
    # If shapes don't match, resize targets to match outputs
    if outputs.shape[2:] != targets.shape[2:]:
        targets = F.interpolate(targets, size=outputs.shape[2:], mode='nearest')
    
    # Ensure targets are float type
    targets = targets.float()
    
    if per_image:
        batch_size = outputs.size(0)
        class_count = outputs.size(1)
        ious = torch.zeros(batch_size, class_count, device=outputs.device)
        
        for i in range(batch_size):
            for c in range(class_count):
                pred = (outputs[i, c] > 0.5).float()
                target = targets[i, c].float()
                
                intersection = (pred * target).sum()
                union = pred.sum() + target.sum() - intersection
                
                iou = (intersection + smooth) / (union + smooth)
                ious[i, c] = iou
                
        # Mean over images
        class_ious = ious.mean(0)
    else:
        # Compute IoU for entire batch
        batch_size = outputs.size(0)
        class_count = outputs.size(1)
        class_ious = torch.zeros(class_count, device=outputs.device)
        
        for c in range(class_count):
            pred = (outputs[:, c] > 0.5).float()
            target = targets[:, c].float()
            
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            
            iou = (intersection + smooth) / (union + smooth)
            class_ious[c] = iou
    
    return class_ious, class_ious.mean()

def compute_dice(outputs, targets, smooth=1e-6, per_image=False):
    """
    Compute Dice coefficient for each class.
    Dice = 2*|X∩Y| / (|X|+|Y|)
    
    Args:
        outputs: Prediction tensor (N, C, H, W) with class probabilities
        targets: Ground truth tensor (N, C, H, W) with binary masks
        smooth: Smoothing factor to avoid division by zero
        per_image: If True, compute Dice for each image and then take the mean
        
    Returns:
        Dice coefficient for each class and mean Dice
    """
    # If shapes don't match, resize targets to match outputs
    if outputs.shape[2:] != targets.shape[2:]:
        targets = F.interpolate(targets, size=outputs.shape[2:], mode='nearest')
    
    # Ensure targets are float type
    targets = targets.float()
    
    if per_image:
        batch_size = outputs.size(0)
        class_count = outputs.size(1)
        dices = torch.zeros(batch_size, class_count, device=outputs.device)
        
        for i in range(batch_size):
            for c in range(class_count):
                pred = (outputs[i, c] > 0.5).float()
                target = targets[i, c].float()
                
                intersection = (pred * target).sum()
                dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
                dices[i, c] = dice
                
        # Mean over images
        class_dices = dices.mean(0)
    else:
        # Compute Dice for entire batch
        batch_size = outputs.size(0)
        class_count = outputs.size(1)
        class_dices = torch.zeros(class_count, device=outputs.device)
        
        for c in range(class_count):
            pred = (outputs[:, c] > 0.5).float()
            target = targets[:, c].float()
            
            intersection = (pred * target).sum()
            dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
            class_dices[c] = dice
    
    return class_dices, class_dices.mean()

def compute_ap(outputs, targets):
    """
    Compute Average Precision (AP) for each class.
    
    Args:
        outputs: Prediction tensor (N, C, H, W) with class probabilities
        targets: Ground truth tensor (N, C, H, W) with binary masks
        
    Returns:
        AP for each class and mean AP
    """
    # If shapes don't match, resize targets to match outputs
    if outputs.shape[2:] != targets.shape[2:]:
        targets = F.interpolate(targets, size=outputs.shape[2:], mode='nearest')
    
    # Ensure targets are float type
    targets = targets.float()
    
    batch_size = outputs.size(0)
    class_count = outputs.size(1)
    aps = torch.zeros(class_count, device=outputs.device)
    
    # Move tensors to CPU for numpy operations
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    for c in range(class_count):
        # Flatten predictions and targets for the current class
        preds = outputs_np[:, c].flatten()
        tgts = targets_np[:, c].flatten()
        
        # Calculate AP using sklearn
        if np.sum(tgts) > 0:  # Only compute AP if there are positive examples
            ap = average_precision_score(tgts, preds)
            aps[c] = ap
    
    return aps, aps.mean()

class MetricTracker:
    """
    Class to keep track of metrics during training and evaluation.
    """
    def __init__(self, lesion_types):
        self.lesion_types = lesion_types
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.iou_per_class = {lesion: [] for lesion in self.lesion_types}
        self.dice_per_class = {lesion: [] for lesion in self.lesion_types}
        self.ap_per_class = {lesion: [] for lesion in self.lesion_types}
        
        self.mean_iou = []
        self.mean_dice = []
        self.mean_ap = []
    
    def update(self, outputs, targets):
        """Update metrics based on a batch of data."""
        # If shapes don't match, resize targets to match outputs
        if outputs.shape[2:] != targets.shape[2:]:
            targets = F.interpolate(targets, size=outputs.shape[2:], mode='nearest')
        
        # Ensure targets are float type
        targets = targets.float()
        
        # Compute IoU
        iou_class, mean_iou = compute_iou(outputs, targets)
        for i, lesion in enumerate(self.lesion_types):
            self.iou_per_class[lesion].append(iou_class[i].item())
        self.mean_iou.append(mean_iou.item())
        
        # Compute Dice
        dice_class, mean_dice = compute_dice(outputs, targets)
        for i, lesion in enumerate(self.lesion_types):
            self.dice_per_class[lesion].append(dice_class[i].item())
        self.mean_dice.append(mean_dice.item())
        
        # Compute AP
        ap_class, mean_ap = compute_ap(outputs, targets)
        for i, lesion in enumerate(self.lesion_types):
            self.ap_per_class[lesion].append(ap_class[i].item())
        self.mean_ap.append(mean_ap.item())
    
    def get_metrics(self):
        """Get the average of all metrics."""
        metrics = {}
        
        # Mean metrics
        metrics['mean_iou'] = np.mean(self.mean_iou)
        metrics['mean_dice'] = np.mean(self.mean_dice)
        metrics['mean_ap'] = np.mean(self.mean_ap)
        
        # Per-class metrics
        for lesion in self.lesion_types:
            metrics[f'iou_{lesion}'] = np.mean(self.iou_per_class[lesion])
            metrics[f'dice_{lesion}'] = np.mean(self.dice_per_class[lesion])
            metrics[f'ap_{lesion}'] = np.mean(self.ap_per_class[lesion])
        
        return metrics