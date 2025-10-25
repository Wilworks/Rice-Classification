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

### Confusion Matrix

```
Confusion Matrix:
[[1599   33]
 [  12 1993]]
```

### Classification Report

```
              precision    recall  f1-score   support

     Class 0       0.99      0.98      0.99      1632
     Class 1       0.98      0.99      0.99      2005

    accuracy                           0.99      3637
   macro avg       0.99      0.99      0.99      3637
weighted avg       0.99      0.99      0.99      3637
```

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
   python inference.py
   ```
   Enter the morphological features when prompted, and the script will output the predicted rice class (Cammeo or Osmancik) along with confidence score.

## Files

- `Rice_Classification.ipynb`: Main notebook with data preprocessing, training, and evaluation
- `train_model.py`: Standalone training script
- `inference.py`: Script for making predictions on new rice samples
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
