# Parkinson's Disease Detection using Hybrid Deep Learning

A deep learning-based system for detecting Parkinson's disease through handwriting analysis, utilizing both wave and spiral drawing patterns. This hybrid model combines ResNet18 and Vision Transformer (ViT) architectures to achieve 96.67% accuracy in early Parkinson's detection.

## Overview

This project implements a multi-modal approach to Parkinson's disease detection by analyzing two types of handwriting patterns:
- **Wave drawings** - analyzed using ResNet18 (CNN-based architecture)
- **Spiral drawings** - analyzed using Vision Transformer (ViT-Base-Patch16-224)

The system uses a hybrid deep learning architecture that fuses features from both modalities through projection layers and a classifier network to provide robust and accurate predictions.

## Dataset Samples

### Wave Drawings

<table>
<tr>
<td width="50%">

**Healthy**

![Healthy Wave](path/to/healthy_wave.png)

</td>
<td width="50%">

**Parkinson's Disease**

![Diseased Wave](path/to/diseased_wave.png)

</td>
</tr>
</table>

### Spiral Drawings

<table>
<tr>
<td width="50%">

**Healthy**

![Healthy Spiral](path/to/healthy_spiral.png)

</td>
<td width="50%">

**Parkinson's Disease**

![Diseased Spiral](path/to/diseased_spiral.png)

</td>
</tr>
</table>

## Model Architecture

The system employs a four-stage training pipeline:

1. **Wave Model Training** (`1_wave_training.py`)
   - Architecture: ResNet18 (pretrained on ImageNet)
   - Input: Grayscale wave drawings (224x224)
   - Output: Binary classification (healthy vs. Parkinson's)
   - Optimizer: AdamW with learning rate 1e-4

2. **Spiral Model Training** (`2_spiral_training.py`)
   - Architecture: Vision Transformer (ViT-Base-Patch16-224, pretrained)
   - Input: Grayscale spiral drawings (224x224)
   - Output: Binary classification (healthy vs. Parkinson's)
   - Optimizer: AdamW with learning rate 1e-4

3. **Feature Fusion** (`3_fusion.py`)
   - ResNet18 features (512-dim) → Projection layer (256-dim)
   - ViT features (768-dim) → Projection layer (256-dim)
   - Concatenated features (512-dim) → Classifier network
   - Classifier: Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→1)

4. **Fine-tuning** (`4_fine_tuning.py`)
   - End-to-end optimization of the hybrid model
   - Differential learning rates: 1e-5 for pretrained branches, 1e-4 for fusion layers
   - Training on paired wave-spiral samples
   - ReduceLROnPlateau scheduler for adaptive learning

## Performance Metrics

The hybrid model achieved exceptional performance on the test dataset:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 96.67% |
| **Precision** | 1.0000 |
| **Recall (Sensitivity)** | 0.9333 |
| **F1-Score** | 0.9655 |

### Class-wise Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Healthy | 0.9375 | 1.0000 | 0.9677 | 15 |
| Parkinson's | 1.0000 | 0.9333 | 0.9655 | 15 |

### Key Highlights

- **Perfect Precision**: Every positive prediction for Parkinson's disease was correct (no false positives)
- **High Recall**: Successfully identified 14 out of 15 Parkinson's cases (93.33% sensitivity)
- **Balanced Performance**: Strong results across both healthy and diseased classes

## Results Visualization

<table>
<tr>
<td width="50%">



## Project Structure

```
.
├── data_preprocessing.ipynb    # Data loading and preprocessing
├── Training/
│   ├── 1_wave_training.py     # Wave pattern model training
│   ├── 2_spiral_training.py   # Spiral pattern model training
│   ├── 3_fusion.py            # Feature fusion implementation
│   └── 4_fine_tuning.py       # Hybrid model fine-tuning
├── Models/
│   ├── wave_model.pth         # Trained wave model weights
│   ├── spiral_model.pth       # Trained spiral model weights
│   └── hybrid_model.pth       # Final hybrid model weights
└── Evaluation/
    ├── evaluate.py            # Model evaluation scripts
    └── interpretation.py      # Results interpretation and visualization
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install required dependencies
pip install torch torchvision timm numpy pandas matplotlib scikit-learn jupyter pillow tqdm pytorch-grad-cam
```

### Key Dependencies
- `torch` & `torchvision` - PyTorch deep learning framework
- `timm` - PyTorch Image Models (for Vision Transformer)
- `pytorch-grad-cam` - For model interpretability (Grad-CAM++)
- `scikit-learn` - For evaluation metrics
- `PIL` (Pillow) - Image processing

## Model Interpretability

The project includes Grad-CAM++ visualization (`interpretation.py`) to understand which regions of the drawings the model focuses on when making predictions. This provides:
- Visual explanations of model decisions
- Identification of discriminative features in wave and spiral patterns
- Increased trust and transparency for clinical applications

## Clinical Significance

The high precision (1.0000) is particularly important in medical diagnosis as it means:
- **No false positives**: Every patient predicted to have Parkinson's actually has the disease
- High confidence in positive predictions
- Reduced unnecessary follow-up tests and patient anxiety

## License

[Add your license here]