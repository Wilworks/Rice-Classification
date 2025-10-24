import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# Set device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

# Model definition (same as in training)
class RiceClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=20):
        super(RiceClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.sigmoid(self.hidden_layer(x))
        return x

# Load the trained model
model = RiceClassifier().to(device)
model.load_state_dict(torch.load('rice_classification_model.pth', map_location=device))
model.eval()

# Load normalization parameters (assuming we saved them or recalculate)
# For simplicity, we'll load the training data to get max values
df = pd.read_csv('riceClassification.csv')
df.dropna(inplace=True)
df.drop(['id'], axis=1, inplace=True)

# Get max values for normalization
max_values = {}
for column in df.columns[:-1]:
    max_values[column] = df[column].abs().max()

def preprocess_input(features):
    """
    Preprocess input features for prediction.

    Args:
        features (dict): Dictionary of feature names and values.

    Returns:
        torch.Tensor: Preprocessed tensor ready for model input.
    """
    # Normalize features
    normalized = []
    for column in df.columns[:-1]:
        if column in features:
            normalized.append(features[column] / max_values[column])
        else:
            raise ValueError(f"Missing feature: {column}")

    return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)

def predict_rice_class(features):
    """
    Predict the class of rice based on input features.

    Args:
        features (dict): Dictionary containing the 10 rice features.

    Returns:
        str: Predicted class ('Cammeo' or 'Osmancik').
    """
    try:
        input_tensor = preprocess_input(features)

        with torch.no_grad():
            output = model(input_tensor).squeeze()
            prediction = 'Cammeo' if output.round().item() == 1 else 'Osmancik'

        confidence = output.item()
        return prediction, confidence
    except Exception as e:
        return f"Error in prediction: {str(e)}", None

if __name__ == "__main__":
    # Example usage
    print("Rice Classification Inference")
    print("Enter the following features (numerical values):")
    features = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
                'Eccentricity', 'Convex_Area', 'Extent', 'Class']

    # Note: 'Class' is the target, but we'll use the first 7 features for prediction
    feature_names = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
                     'Eccentricity', 'Convex_Area', 'Extent']

    input_features = {}
    for feature in feature_names:
        value = float(input(f"Enter {feature}: "))
        input_features[feature] = value

    prediction, confidence = predict_rice_class(input_features)
    if confidence is not None:
        print(f"Predicted class: {prediction}")
        print(".3f")
    else:
        print(prediction)
