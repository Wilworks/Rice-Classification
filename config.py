import torch

# Model configuration
INPUT_SIZE = 10
HIDDEN_SIZE = 20
OUTPUT_SIZE = 1

# Class names
CLASS_NAMES = ['Class 0', 'Class 1']

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
