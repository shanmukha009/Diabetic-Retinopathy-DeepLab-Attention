"""
Inference script for applying trained DR lesion segmentation models to new images.
"""

import os
import argparse
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Local imports
import config
from models.deeplab import get_deeplab_model
from models.attention_unet import get_attention_unet
from utils.visualization import (
    create_mask_overlay, 
    create_composite_visualization,
    save_visualization
)

def parse_args():
    parser = argparse.ArgumentParser(description='Apply DR lesion segmentation model to new images')
    parser.add_argument('--model', type=str, default='deeplab', choices=['deeplab', 'attention_unet'],
                        help='Model architecture to use')
    parser.add_argument('--encoder', type=str, default='resnet50', 
                        help='Encoder backbone (for DeepLab)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU ID(s) to use, comma-separated')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for segmentation')
    parser.add_argument('--save_overlay', action='store_true',
                        help='Save overlay visualization')
    parser.add_argument('--save_composite', action='store_true',
                        help='Save composite visualization with confidence maps')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save individual binary masks for each lesion type')
    return parser.parse_args()

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Preprocess an image for inference.
    
    Args:
        image_path: Path to the image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed tensor
    """
    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    
    return input_tensor, image

def run_inference(model, image_path, device, confidence_threshold=0.5):
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image
        device: Device to run inference on
        confidence_threshold: Confidence threshold for segmentation
        
    Returns:
        Original image, preprocessed tensor, predictions
    """
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)
    
    # Add batch dimension and move to device
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output)
        
        # Apply threshold to get binary masks
        masks = (probs > confidence_threshold).float()
    
    return original_image, input_tensor, probs[0], masks[0]

def save_results(original_image, input_tensor, probs, masks, output_dir, image_name, args):
    """
    Save inference results.
    
    Args:
        original_image: Original PIL image
        input_tensor: Preprocessed tensor
        probs: Probability maps for each lesion type
        masks: Binary masks for each lesion type
        output_dir: Directory to save results
        image_name: Base name for output files
        args: Command line arguments
    """
    # Convert input tensor to numpy for visualization
    input_np = input_tensor.cpu().numpy().transpose(1, 2, 0)
    input_np = input_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    input_np = np.clip(input_np, 0, 1)
    
    # Save overlay if requested
    if args.save_overlay:
        # Create overlay image
        overlay = create_mask_overlay(input_np, masks.cpu().numpy(), config.LESION_TYPES)
        
        # Save overlay
        overlay_path = os.path.join(output_dir, f"{image_name}_overlay.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # Save composite visualization if requested
    if args.save_composite:
        # Create composite visualization
        composite = create_composite_visualization(input_tensor, probs, config.LESION_TYPES)
        
        # Save composite
        composite_path = os.path.join(output_dir, f"{image_name}_composite.png")
        save_visualization(composite, composite_path)
    
    # Save individual masks if requested
    if args.save_masks:
        masks_dir = os.path.join(output_dir, f"{image_name}_masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        # Save masks for each lesion type
        for i, lesion_type in enumerate(config.LESION_TYPES):
            mask = masks[i].cpu().numpy()
            mask_path = os.path.join(masks_dir, f"{lesion_type}.png")
            plt.figure(figsize=(10, 10))
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
            plt.close()

def main():
    """
    Main inference function.
    """
    args = parse_args()
    
    # Set CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    num_classes = len(config.LESION_TYPES)
    if args.model == 'deeplab':
        model = get_deeplab_model(
            num_classes=num_classes,
            backbone=args.encoder,
            pretrained=False  # No need for pretrained weights when loading checkpoint
        )
    elif args.model == 'attention_unet':
        model = get_attention_unet(
            num_classes=num_classes,
            pretrained_encoder=False  # No need for pretrained weights when loading checkpoint
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Get input image paths
    if os.path.isdir(args.input):
        # Process all images in directory
        image_paths = glob.glob(os.path.join(args.input, '*.jpg')) + \
                      glob.glob(os.path.join(args.input, '*.jpeg')) + \
                      glob.glob(os.path.join(args.input, '*.png'))
    else:
        # Process a single image
        image_paths = [args.input]
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Get image name
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Run inference
        original_image, input_tensor, probs, masks = run_inference(
            model, image_path, device, args.confidence_threshold
        )
        
        # Save results
        save_results(
            original_image, input_tensor, probs, masks, output_dir, image_name, args
        )
    
    print(f"Inference completed! Results saved to {output_dir}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    main()