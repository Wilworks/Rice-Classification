import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# Define the model architecture (same as in training)
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # Fully Connected layer
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # Forward Propagation
    def forward(self, X):
        # First pass
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        # Second Pass
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)

        # Review to flatten
        X = X.view(-1, 16*5*5)

        # Fully Connected Layer
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = ConvolutionalNetwork().to(device)
model.load_state_dict(torch.load('digits_cnn_model.pth', map_location=device))
model.eval()

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((28, 28)),  # MNIST size
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

def predict_digit(image_path):
    """
    Predict the digit from an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        int: Predicted digit (0-9).
    """
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            predicted = torch.argmax(output, dim=1).item()

        return predicted
    except Exception as e:
        return f"Error predicting digit: {str(e)}"

def predict_from_array(image_array):
    """
    Predict digit from a numpy array (28x28 grayscale).

    Args:
        image_array (np.ndarray): 28x28 numpy array.

    Returns:
        int: Predicted digit (0-9).
    """
    try:
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0).to(device)
        # Normalize
        image_tensor = (image_tensor - 0.1307) / 0.3081

        with torch.no_grad():
            output = model(image_tensor)
            predicted = torch.argmax(output, dim=1).item()

        return predicted
    except Exception as e:
        return f"Error predicting from array: {str(e)}"

if __name__ == "__main__":
    # Example usage
    image_path = input("Enter the path to a digit image (28x28 grayscale): ")
    if image_path:
        prediction = predict_digit(image_path)
        print(f"Predicted digit: {prediction}")
    else:
        print("No image path provided.")

    # For testing with MNIST test data
    print("\nTesting with a few MNIST test samples...")
    test_data = datasets.MNIST(root='cnn_data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    model.eval()
    correct = 0
    total = 5  # Test 5 samples

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= total:
                break
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
            print(f"Sample {i+1}: Predicted {predicted.item()}, Actual {labels.item()}")

    print(f"\nAccuracy on {total} test samples: {100 * correct / total:.2f}%")
