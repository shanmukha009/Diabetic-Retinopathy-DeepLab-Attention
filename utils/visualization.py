"""
Visualization utilities for DR lesion segmentation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2

# Define custom colors for different lesion types
LESION_COLORS = {
    'MA': [1.0, 0.0, 0.0],  # Red for microaneurysms
    'EX': [1.0, 1.0, 0.0],  # Yellow for hard exudates
    'SE': [0.0, 1.0, 1.0],  # Cyan for soft exudates
    'HE': [0.0, 0.0, 1.0]   # Blue for hemorrhages
}

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor image with given mean and standard deviation.
    
    Args:
        tensor: Tensor image of size (C, H, W)
        mean: Mean for each channel
        std: Standard deviation for each channel
    
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    import torch
    denorm_tensor = tensor.clone()
    
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clamp to ensure values are in [0, 1]
    return torch.clamp(denorm_tensor, 0, 1)

def create_mask_overlay(image, mask, lesion_types, alpha=0.5):
    """
    Create an overlay image with masks colored by lesion type.
    
    Args:
        image: Numpy array of image (H, W, 3) with values in [0, 1]
        mask: Numpy array of masks (num_classes, H, W) with values in [0, 1]
        lesion_types: List of lesion types corresponding to mask channels
        alpha: Transparency factor for overlay
    
    Returns:
        Overlay image as numpy array (H, W, 3)
    """
    # Initialize overlay with the original image
    overlay = image.copy()
    
    # Handle dimension mismatches by resizing the mask to match the image
    img_h, img_w = image.shape[:2]
    mask_h, mask_w = mask.shape[1:3]
    
    # If dimensions don't match, resize the mask to match the image
    if img_h != mask_h or img_w != mask_w:
        import torch
        import torch.nn.functional as F
        
        # Convert mask to torch tensor for resizing
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Add batch dimension
        
        # Resize mask to match image dimensions
        mask_tensor = F.interpolate(mask_tensor, size=(img_h, img_w), mode='nearest')
        
        # Convert back to numpy and remove batch dimension
        mask = mask_tensor.squeeze(0).numpy()
    
    # Create a binary mask to track where any lesion has been detected
    any_lesion = np.zeros((img_h, img_w), dtype=bool)
    
    # Apply each mask with its corresponding color
    for i, lesion_type in enumerate(lesion_types):
        if lesion_type in LESION_COLORS:
            # Threshold the mask
            binary_mask = mask[i] > 0.5
            
            # Skip if no lesion is detected
            if not np.any(binary_mask):
                continue
            
            # Update the overlay for this lesion type
            color = np.array(LESION_COLORS[lesion_type])
            for c in range(3):
                overlay[..., c] = np.where(
                    binary_mask & ~any_lesion,
                    (1 - alpha) * overlay[..., c] + alpha * color[c],
                    overlay[..., c]
                )
            
            # Update the any_lesion mask
            any_lesion = any_lesion | binary_mask
    
    return overlay

def visualize_predictions(images, masks, predictions, lesion_types):
    """
    Visualize model predictions alongside ground truth masks.
    
    Args:
        images: Tensor of images (B, C, H, W)
        masks: Tensor of ground truth masks (B, num_classes, H, W)
        predictions: Tensor of predicted masks (B, num_classes, H, W)
        lesion_types: List of lesion types corresponding to mask channels
    
    Returns:
        Visualization image as numpy array (H, W*3, 3)
    """
    import torch
    
    # Convert tensors to numpy arrays
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    # Ensure masks and predictions have the same dimensions as images
    batch_size, _, img_h, img_w = images_np.shape
    _, num_classes, mask_h, mask_w = masks_np.shape
    
    # If dimensions don't match, resize masks and predictions
    if img_h != mask_h or img_w != mask_w:
        masks_tensor = torch.from_numpy(masks_np)
        masks_tensor = torch.nn.functional.interpolate(masks_tensor, size=(img_h, img_w), mode='nearest')
        masks_np = masks_tensor.numpy()
    
    _, _, pred_h, pred_w = predictions_np.shape
    if img_h != pred_h or img_w != pred_w:
        predictions_tensor = torch.from_numpy(predictions_np)
        predictions_tensor = torch.nn.functional.interpolate(predictions_tensor, size=(img_h, img_w), mode='nearest')
        predictions_np = predictions_tensor.numpy()
    
    # Create a figure for visualization
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    
    # Handle the case when batch_size is 1
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i in range(batch_size):
        # Denormalize and transpose the image to (H, W, C)
        image = images_np[i].transpose(1, 2, 0)
        
        # Apply denormalization
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        
        # Get masks and predictions for this image
        mask = masks_np[i]
        prediction = predictions_np[i]
        
        # Create overlays
        mask_overlay = create_mask_overlay(image, mask, lesion_types)
        pred_overlay = create_mask_overlay(image, prediction, lesion_types)
        
        # Plot images
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_overlay)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    # Adjust layout and convert to numpy array
    plt.tight_layout()
    
    # Convert to numpy array - FIXED with multiple methods for different matplotlib versions
    fig.canvas.draw()
    try:
        # Try the newer buffer_rgba method first
        buffer = fig.canvas.buffer_rgba()
        vis_image = np.asarray(buffer)[:, :, :3]  # Remove alpha channel
    except (AttributeError, ValueError):
        try:
            # Try alternative method with tostring_argb
            data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            vis_image = data[:, :, 1:4]  # ARGB to RGB (skip alpha channel)
        except (AttributeError, ValueError):
            # Last resort: use savefig with BytesIO
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = plt.imread(buf)
            vis_image = (img * 255).astype(np.uint8)
            if vis_image.shape[2] > 3:
                vis_image = vis_image[:, :, :3]  # Remove alpha if present
            buf.close()
    
    # Close the figure to free memory
    plt.close(fig)
    
    return vis_image

def save_visualization(vis_image, save_path):
    """
    Save visualization image to disk.
    
    Args:
        vis_image: Numpy array of visualization image
        save_path: Path to save the image
    """
    cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

def create_lesion_legend():
    """
    Create a legend image showing the color coding for different lesion types.
    
    Returns:
        Legend image as numpy array
    """
    fig, ax = plt.subplots(figsize=(6, 2))
    
    # Create patches for each lesion type
    for i, (lesion_type, color) in enumerate(LESION_COLORS.items()):
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.9, color=color))
        ax.text(i + 0.45, 0.45, lesion_type, ha='center', va='center')
    
    ax.set_xlim(0, len(LESION_COLORS))
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Lesion Types')
    
    # Convert to numpy array - FIXED with multiple methods for different matplotlib versions
    fig.canvas.draw()
    try:
        # Try the newer buffer_rgba method first
        buffer = fig.canvas.buffer_rgba()
        legend_image = np.asarray(buffer)[:, :, :3]  # Remove alpha channel
    except (AttributeError, ValueError):
        try:
            # Try alternative method with tostring_argb
            data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            legend_image = data[:, :, 1:4]  # ARGB to RGB (skip alpha channel)
        except (AttributeError, ValueError):
            # Last resort: use savefig with BytesIO
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = plt.imread(buf)
            legend_image = (img * 255).astype(np.uint8)
            if legend_image.shape[2] > 3:
                legend_image = legend_image[:, :, :3]  # Remove alpha if present
            buf.close()
    
    # Close the figure to free memory
    plt.close(fig)
    
    return legend_image

def create_confidence_map(prediction, lesion_type_idx, cmap='jet'):
    """
    Create a confidence map visualization for a specific lesion type.
    
    Args:
        prediction: Prediction tensor for a single image (num_classes, H, W)
        lesion_type_idx: Index of the lesion type to visualize
        cmap: Colormap to use
    
    Returns:
        Confidence map as numpy array (H, W, 3)
    """
    # Get the prediction for the specified lesion type
    lesion_pred = prediction[lesion_type_idx].cpu().numpy()
    
    # Create a colormap
    colormap = plt.get_cmap(cmap)
    
    # Apply colormap
    confidence_map = colormap(lesion_pred)[:, :, :3]
    
    return confidence_map

def create_composite_visualization(image, predictions, lesion_types):
    """
    Create a composite visualization with original image and confidence maps for each lesion type.
    
    Args:
        image: Tensor of a single image (C, H, W)
        predictions: Tensor of predicted masks for a single image (num_classes, H, W)
        lesion_types: List of lesion types corresponding to mask channels
    
    Returns:
        Composite visualization as numpy array
    """
    import torch
    
    # Convert image to numpy array and transpose to (H, W, C)
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    
    # Apply denormalization
    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image_np = np.clip(image_np, 0, 1)
    
    # Ensure predictions have the same dimensions as the image
    img_h, img_w = image_np.shape[:2]
    pred_h, pred_w = predictions.shape[1:]
    
    # If dimensions don't match, resize predictions to match the image
    if img_h != pred_h or img_w != pred_w:
        pred_tensor = predictions.unsqueeze(0)  # Add batch dimension
        pred_tensor = torch.nn.functional.interpolate(pred_tensor, size=(img_h, img_w), mode='nearest')
        predictions = pred_tensor.squeeze(0)  # Remove batch dimension
    
    # Create a figure
    fig, axes = plt.subplots(1, len(lesion_types) + 1, figsize=(5 * (len(lesion_types) + 1), 5))
    
    # Plot original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot confidence map for each lesion type
    for i, lesion_type in enumerate(lesion_types):
        confidence_map = create_confidence_map(predictions, i)
        axes[i + 1].imshow(confidence_map)
        axes[i + 1].set_title(f'{lesion_type} Confidence')
        axes[i + 1].axis('off')
    
    # Adjust layout and convert to numpy array
    plt.tight_layout()
    
    # Convert to numpy array - FIXED with multiple methods for different matplotlib versions
    fig.canvas.draw()
    try:
        # Try the newer buffer_rgba method first
        buffer = fig.canvas.buffer_rgba()
        vis_image = np.asarray(buffer)[:, :, :3]  # Remove alpha channel
    except (AttributeError, ValueError):
        try:
            # Try alternative method with tostring_argb
            data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            vis_image = data[:, :, 1:4]  # ARGB to RGB (skip alpha channel)
        except (AttributeError, ValueError):
            # Last resort: use savefig with BytesIO
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = plt.imread(buf)
            vis_image = (img * 255).astype(np.uint8)
            if vis_image.shape[2] > 3:
                vis_image = vis_image[:, :, :3]  # Remove alpha if present
            buf.close()
    
    # Close the figure to free memory
    plt.close(fig)
    
    return vis_image