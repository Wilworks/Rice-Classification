import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Set device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

# Load and preprocess data
df = pd.read_csv('riceClassification.csv')
df.dropna(inplace=True)
df.drop(['id'], axis=1, inplace=True)

# Normalize features
for column in df.columns[:-1]:
    df[column] = df[column] / df[column].abs().max()

X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=41)

# Dataset class
class RiceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32).to(device)
        self.y = torch.tensor(y.values, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Create datasets and dataloaders
train_dataset = RiceDataset(X_train, y_train)
val_dataset = RiceDataset(X_val, y_val)
test_dataset = RiceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Model definition
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

# Initialize model
model = RiceClassifier().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += ((outputs.round() == labels).sum().item())

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += ((outputs.round() == labels).sum().item())

    train_acc = train_correct / len(train_dataset) * 100
    val_acc = val_correct / len(val_dataset) * 100

    print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Acc: {val_acc:.2f}%')

# Test evaluation
model.eval()
test_loss = 0
test_correct = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        test_correct += ((outputs.round() == labels).sum().item())

test_acc = test_correct / len(test_dataset) * 100
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_acc:.2f}%')

# Save model
torch.save(model.state_dict(), 'rice_classification_model.pth')
print('Model saved to rice_classification_model.pth')
