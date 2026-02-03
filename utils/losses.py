"""
Custom loss functions for DR lesion segmentation.
Focusing on:
1. Class imbalance (few lesion pixels, many background pixels)
2. Small lesions like microaneurysms
3. Boundary detection for accurate lesion delineation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    Handles class imbalance better than cross-entropy.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model output tensor (N, C, H, W)
            targets: Ground truth tensor (N, C, H, W)
            
        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        outputs_sigmoid = torch.sigmoid(outputs)
        
        # If shapes don't match, resize targets to match outputs
        if outputs.shape[2:] != targets.shape[2:]:
            print(f"Resizing targets from {targets.shape[2:]} to {outputs.shape[2:]}")
            targets = F.interpolate(targets, size=outputs.shape[2:], mode='nearest')
        
        # Ensure targets are float type
        targets = targets.float()
        
        # Calculate Dice for each class
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        loss = 0
        
        # Process each class separately to avoid dimension issues
        for i in range(num_classes):
            # Get class-specific outputs and targets
            pred = outputs_sigmoid[:, i].reshape(-1)  # Flatten to 1D using reshape
            target = targets[:, i].reshape(-1)        # Flatten to 1D using reshape
            
            # Calculate intersection and union
            intersection = (pred * target).sum()
            pred_sum = pred.sum()
            target_sum = target.sum()
            
            # Calculate Dice coefficient
            dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            
            # Dice loss
            loss += (1 - dice)
        
        # Return mean loss over all classes
        return loss / num_classes

class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy with logits loss with class weights.
    """
    def __init__(self, pos_weights=None):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weights = pos_weights
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model output tensor (N, C, H, W)
            targets: Ground truth tensor (N, C, H, W)
            
        Returns:
            Weighted BCE loss value
        """
        # If shapes don't match, resize targets to match outputs
        if outputs.shape[2:] != targets.shape[2:]:
            targets = F.interpolate(targets, size=outputs.shape[2:], mode='nearest')
        
        # Ensure targets are float type
        targets = targets.float()
        
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        
        loss = 0
        for c in range(num_classes):
            # Get the current class outputs and targets
            class_outputs = outputs[:, c]  # Shape: [N, H, W]
            class_targets = targets[:, c]  # Shape: [N, H, W]
            
            # Apply class weight if available
            pos_weight = self.pos_weights[c] if self.pos_weights is not None else 1.0
            pos_weight_tensor = torch.tensor([pos_weight], device=outputs.device)
            
            # Calculate binary cross-entropy for this class
            class_loss = F.binary_cross_entropy_with_logits(
                class_outputs, 
                class_targets, 
                pos_weight=pos_weight_tensor,
                reduction='mean'
            )
            
            loss += class_loss
        
        return loss / num_classes

class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    Focuses more on hard-to-classify examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model output tensor (N, C, H, W)
            targets: Ground truth tensor (N, C, H, W)
            
        Returns:
            Focal loss value
        """
        # If shapes don't match, resize targets to match outputs
        if outputs.shape[2:] != targets.shape[2:]:
            targets = F.interpolate(targets, size=outputs.shape[2:], mode='nearest')
        
        # Ensure targets are float type
        targets = targets.float()
        
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        
        # Apply sigmoid to outputs
        probs = torch.sigmoid(outputs)
        
        # Calculate focal loss
        loss = 0
        for c in range(num_classes):
            # Get class-specific outputs, targets, and probabilities
            class_targets = targets[:, c]  # Shape: [N, H, W]
            class_probs = probs[:, c]      # Shape: [N, H, W]
            
            # Calculate pt (probability of being the true class)
            pt = torch.where(class_targets == 1, class_probs, 1 - class_probs)
            
            # Calculate alpha factor
            alpha_factor = torch.where(
                class_targets == 1, 
                torch.ones_like(class_targets) * self.alpha,
                torch.ones_like(class_targets) * (1 - self.alpha)
            )
            
            # Calculate modulating factor
            modulating_factor = (1.0 - pt) ** self.gamma
            
            # Calculate focal loss for this class
            class_loss = -alpha_factor * modulating_factor * torch.log(torch.clamp(pt, 1e-8, 1.0))
            
            if self.reduction == 'mean':
                class_loss = class_loss.mean()
            elif self.reduction == 'sum':
                class_loss = class_loss.sum()
            
            loss += class_loss
        
        return loss / num_classes

class BoundaryLoss(nn.Module):
    """
    Boundary loss for accurate lesion boundary detection.
    Especially useful for small lesions like microaneurysms.
    """
    def __init__(self, theta=1.5):
        super(BoundaryLoss, self).__init__()
        self.theta = theta
        
        # We'll create the Sobel filters in the forward pass
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model output tensor (N, C, H, W)
            targets: Ground truth tensor (N, C, H, W)
            
        Returns:
            Boundary loss value
        """
        # If shapes don't match, resize targets to match outputs
        if outputs.shape[2:] != targets.shape[2:]:
            targets = F.interpolate(targets, size=outputs.shape[2:], mode='nearest')
        
        # Ensure targets are float type
        targets = targets.float()
        
        # Create Sobel filters directly on the same device as inputs
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                              dtype=torch.float32, device=outputs.device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                              dtype=torch.float32, device=outputs.device)
        
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        
        probs = torch.sigmoid(outputs)
        loss = 0
        
        for c in range(num_classes):
            # Get class-specific targets and probabilities
            class_targets = targets[:, c:c+1]  # Keep channel dim for convolution: [N, 1, H, W]
            class_probs = probs[:, c:c+1]      # Keep channel dim: [N, 1, H, W]
            
            # Calculate gradient magnitude of targets (boundary)
            pad = 1
            grad_x = F.conv2d(class_targets, sobel_x, padding=pad)
            grad_y = F.conv2d(class_targets, sobel_y, padding=pad)
            
            # Calculate gradient magnitude
            target_boundary = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
            
            # Calculate weighted distance to boundary
            boundary_weight = torch.exp(-self.theta * target_boundary)
            
            # Calculate binary cross-entropy with boundary weight
            bce = F.binary_cross_entropy(
                class_probs, 
                class_targets,
                reduction='none'
            )
            
            weighted_bce = (boundary_weight * bce).mean()
            loss += weighted_bce
        
        return loss / num_classes

class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions for optimal performance.
    """
    def __init__(self, lesion_types, weights=None):
        super(CombinedLoss, self).__init__()
        self.lesion_types = lesion_types
        self.num_classes = len(lesion_types)
        
        # Define lesion-specific weights based on paper findings
        self.lesion_weights = torch.tensor([config.LESION_WEIGHTS[lesion] for lesion in lesion_types])
        
        # Initialize component loss functions
        self.dice_loss = DiceLoss()
        self.bce_loss = WeightedBCEWithLogitsLoss(pos_weights=self.lesion_weights)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.boundary_loss = BoundaryLoss(theta=1.5)
        
        # Weights for each loss component
        if weights is None:
            self.weights = {
                'dice': 1.0,
                'bce': 0.5,
                'focal': 1.0,
                'boundary': 0.5
            }
        else:
            self.weights = weights
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model output tensor (N, C, H, W)
            targets: Ground truth tensor (N, C, H, W)
            
        Returns:
            Combined loss value
        """
        # If targets has channels in the wrong dimension, fix it
        if len(targets.shape) == 4 and targets.size(1) != outputs.size(1) and targets.size(-1) == outputs.size(1):
            targets = targets.permute(0, 3, 1, 2)
        
        # Ensure targets are float type
        targets = targets.float()
        
        dice = self.dice_loss(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        focal = self.focal_loss(outputs, targets)
        boundary = self.boundary_loss(outputs, targets)
        
        combined_loss = (
            self.weights['dice'] * dice +
            self.weights['bce'] * bce +
            self.weights['focal'] * focal +
            self.weights['boundary'] * boundary
        )
        
        return combined_loss