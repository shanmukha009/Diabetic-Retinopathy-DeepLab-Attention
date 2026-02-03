"""
Configuration parameters for the DDR lesion segmentation project.
"""

import os
import torch

# Dataset parameters
DATA_ROOT = "/content/drive/MyDrive/DDR/DDR-dataset/lesion_segmentation"
LESION_TYPES = ['MA', 'EX', 'SE', 'HE']
IMG_SIZE = (512, 512)  # Resize all images to this size

# Training parameters
BATCH_SIZE = 4
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special attention to microaneurysms (MA) which performed poorly in the baseline
# Based on the paper, MA has the lowest AP and IoU scores
LESION_WEIGHTS = {
    'MA': 4.0,  # Higher weight for the worst performing class
    'EX': 1.0,
    'SE': 1.5,
    'HE': 2.0
}

# Model parameters
ENCODER_NAME = "resnet50"  # Backbone for DeepLab and other models
PRETRAINED = True

# Augmentation parameters
AUGMENTATION_PROB = 0.5

# Optimization
OPTIMIZER = "Adam"
SCHEDULER = "ReduceLROnPlateau"
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# Output directories
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)