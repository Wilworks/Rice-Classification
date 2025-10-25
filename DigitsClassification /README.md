# MNIST Digits Classification with CNN

A Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset.

## Overview

This project implements a CNN model to classify handwritten digits (0-9) from the MNIST dataset. The model uses PyTorch and achieves high accuracy on both training and test sets.

## Dataset

The MNIST dataset contains:
- **Training set**: 60,000 grayscale images (28x28 pixels)
- **Test set**: 10,000 grayscale images (28x28 pixels)
- **Classes**: 10 (digits 0-9)
- **Image format**: Grayscale, normalized to [0,1]

## Model Architecture

**Convolutional Neural Network:**
- **Conv1**: 6 filters, 3x3 kernel, stride 1 → ReLU → MaxPool 2x2
- **Conv2**: 16 filters, 3x3 kernel, stride 1 → ReLU → MaxPool 2x2
- **FC1**: 400 → 120 neurons → ReLU
- **FC2**: 120 → 84 neurons → ReLU
- **FC3**: 84 → 10 neurons → LogSoftmax

## Performance

- **Training Accuracy**: ~99.10%
- **Test Accuracy**: ~98.76%
- **Training Time**: ~1 minute (5 epochs)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## Usage

1. **Training**:
   ```python
   # Run the Jupyter notebook Pipeline.ipynb
   # or execute the training script
   python train_cnn.py
   ```

2. **Inference**:
   ```python
   python inference.py
   ```
   Provide the path to a 28x28 grayscale digit image, and the script will output the predicted digit (0-9).

## Training Details

- **Batch Size**: 10
- **Epochs**: 5
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (lr=0.001)
- **Device**: MPS (Apple Silicon) or CPU

## Files

- `Pipeline.ipynb`: Main notebook with data loading, model training, and evaluation
- `train_cnn.py`: Standalone training script
- `inference.py`: Script for making predictions on new digit images
- `digits_cnn_model.pth`: Trained model weights
- `cnn_data/`: MNIST dataset (downloaded automatically)
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Data Preprocessing

- Images converted to tensors and normalized to [0,1]
- No additional preprocessing required (MNIST is pre-processed)

## Results

The CNN model achieves excellent performance on the MNIST dataset, demonstrating effective feature extraction and classification capabilities for handwritten digit recognition.

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests for improvements.
