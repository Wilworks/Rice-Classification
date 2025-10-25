import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load and preprocess the rice classification dataset"""
    df = pd.read_csv(filepath)
    X = df.drop(['id', 'Class'], axis=1).values if 'id' in df.columns else df.drop('Class', axis=1).values
    y = df['Class'].values

    # Normalize by max absolute value (as in original)
    X_normalized = X / np.max(np.abs(X), axis=0)

    return X_normalized, y

def create_model():
    """Create the rice classification model"""
    class RiceClassifier(torch.nn.Module):
        def __init__(self):
            super(RiceClassifier, self).__init__()
            self.input_layer = torch.nn.Linear(10, 20)
            self.relu = torch.nn.ReLU()
            self.linear = torch.nn.Linear(20, 1)

        def forward(self, x):
            x = self.input_layer(x)
            x = self.relu(x)
            x = self.linear(x)
            return x

    return RiceClassifier()

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix using seaborn"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load test data
    X_test, y_test = load_data('riceClassification.csv')

    # Create test dataset
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = create_model()
    model.load_state_dict(torch.load('rice_classification_model.pth'))
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).squeeze().long()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Results
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))

    # Plot confusion matrix
    plot_confusion_matrix(cm, ['Class 0', 'Class 1'], "Rice Classification Confusion Matrix")

if __name__ == "__main__":
    main()
