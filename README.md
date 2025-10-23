# Rice Classification Model

A PyTorch-based neural network for classifying rice types using morphological features.

## Overview

This project implements a binary classification model to distinguish between two rice varieties (Arborio and Basmati) based on physical characteristics such as area, perimeter, eccentricity, and other morphological features.

## Dataset

The dataset contains 18,185 rice grain samples with 11 features:
- Area
- MajorAxisLength
- MinorAxisLength
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Perimeter
- Roundness
- AspectRation
- Class (target: 0 or 1)

## Model Architecture

- **Input Layer**: 10 features (after preprocessing)
- **Hidden Layer**: 20 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation
- **Loss Function**: Binary Cross-Entropy Loss
- **Optimizer**: Adam (lr=0.001)

## Performance

- **Training Accuracy**: ~98.74%
- **Validation Accuracy**: ~98.10%
- **Test Accuracy**: ~98.63%

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Installation

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## Usage

1. **Training**:
   ```python
   # Run the Jupyter notebook Rice_Classification.ipynb
   # or execute the training script
   python train_model.py
   ```

2. **Inference**:
   ```python
   import torch
   from model import Rice

   # Load model
   model = Rice()
   model.load_state_dict(torch.load('rice_classification_model.pth'))
   model.eval()

   # Make predictions
   # ... (add inference code)
   ```

## Files

- `Rice_Classification.ipynb`: Main notebook with data preprocessing, training, and evaluation
- `rice_classification_model.pth`: Trained model weights
- `riceClassification.csv`: Dataset
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Data Preprocessing

- Removed 'id' column
- Normalized features by dividing by maximum absolute value
- Split data: 70% train, 15% validation, 15% test

## Training Details

- Batch size: 32
- Epochs: 10
- Device: MPS (Apple Silicon) or CPU

## Results

The model achieves high accuracy on both validation and test sets, demonstrating effective classification of rice types based on morphological features.

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests for improvements.
