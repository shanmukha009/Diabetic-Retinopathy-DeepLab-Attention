# Diabetic Retinopathy Lesion Segmentation through Attention Mechanisms

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

Attention-enhanced DeepLab-V3+ architecture for automated diabetic retinopathy lesion segmentation, achieving significant improvements in early detection and clinical screening capabilities.

**Key Achievement:** 272% improvement in microaneurysm detection accuracy, enabling scalable diabetic eye disease screening.

## Author

**Shanmukha Raj**  
San Jose State University  
Master's in Applied Data Intelligence (Graduated Dec 2025)

## Highlights

- **272% improvement** in microaneurysm detection (0.0763 vs. 0.0205 AP)
- **10.5% increase** in mean Average Precision across all lesion types
- **134% improvement** in hemorrhage detection
- Clinically significant breakthrough for early DR screening

## Technical Approach

### Architecture
- **Base Model:** DeepLab-V3+ with ResNet-50 backbone
- **Enhancement:** Convolutional Block Attention Module (CBAM) integration
- **Innovation:** Attention-guided feature refinement for precise pixel-level segmentation

### Lesion Types Detected
1. Microaneurysms
2. Hemorrhages
3. Hard Exudates
4. Soft Exudates (Cotton Wool Spots)

## Technologies Used

- **Framework:** PyTorch 2.0+
- **Architecture:** DeepLab-V3+ with CBAM
- **Language:** Python 3.8+
- **Key Libraries:** torchvision, numpy, opencv-python, matplotlib

## Project Structure
```
├── models/
│   ├── deeplab.py          # DeepLab-V3+ implementation with CBAM
│   └── unet.py             # U-Net baseline for comparison
├── data/
│   ├── dataloader.py       # Custom data loading pipeline
│   └── augmentation.py     # Medical image augmentation
├── utils/
│   ├── losses.py           # Combined loss functions
│   ├── metrics.py          # Evaluation metrics (mAP, IoU)
│   └── visualization.py    # Result visualization tools
├── train.py                # Training pipeline
├── evaluate.py             # Evaluation script
└── inference.py            # Inference on new images
```

## Key Features

 Attention-enhanced feature extraction  
 Multi-scale lesion detection  
 Pixel-level segmentation accuracy  
 Clinically actionable insights  
 Scalable screening solution

## Results

| Metric | Improvement |
|--------|-------------|
| Microaneurysm Detection | +272% |
| Mean Average Precision | +10.5% |
| Hemorrhage Detection | +134% |

## Clinical Impact

This system enables:
- Early detection of diabetic retinopathy
- Scalable screening programs for underserved populations
- Clinically actionable insights for ophthalmologists
- Reduced diagnostic burden through automated pre-screening

## Installation
```bash
# Clone repository
git clone https://github.com/shanmukha009/Diabetic-Retinopathy-DeepLab-Attention.git
cd Diabetic-Retinopathy-DeepLab-Attention

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py --config config.py
```

### Evaluation
```bash
python evaluate.py --model_path checkpoints/best_model.pth
```

### Inference
```bash
python inference.py --image_path path/to/fundus_image.jpg
```

## Technical Implementation Details

### Model Architecture
The implementation uses DeepLab-V3+ as the backbone with strategic CBAM attention module integration at multiple scales to enhance lesion-specific feature extraction.

### Training Strategy
- Mixed precision training for efficiency
- Custom weighted loss function for class imbalance
- Data augmentation specific to medical imaging
- Cross-validation for robust evaluation

### Evaluation Metrics
- Mean Average Precision (mAP)
- Intersection over Union (IoU)
- Per-lesion precision and recall
- Clinical relevance scoring

## Future Improvements

- Integration with clinical workflow systems
- Real-time inference optimization
- Multi-modal data fusion (OCT + fundus imaging)
- Explainable AI for clinical trust

## License

MIT License

## Contact

**Shanmukha Raj**  
 shanmukhraj965@gmail.com  
 [LinkedIn](https://www.linkedin.com/in/shanmukha-raj/)  
 [GitHub](https://github.com/shanmukha009)

---

*Graduate Research Project | San Jose State University | December 2024*