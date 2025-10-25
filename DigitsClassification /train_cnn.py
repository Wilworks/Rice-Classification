import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Convert MNIST image files into tensors of 4D. No of images, height, width and colour channels
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='cnn_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='cnn_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

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

# Model Instance
torch.manual_seed(41)
model = ConvolutionalNetwork()
print('Model created successfully')

# Loss Function Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()

# Trackers
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    epoch_train_loss = 0

    # TRAIN
    model.train()
    for b, (X_train, Y_train) in enumerate(train_loader):
        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)

        epoch_train_loss += loss.item()
        predicted = torch.max(y_pred.data, 1)[1]
        trn_corr += (predicted == Y_train).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (b + 1) % 600 == 0:
            print(f'Epoch: {i+1} Batch: {b+1} Loss: {loss.item():.4f}')

    train_losses.append(epoch_train_loss / len(train_loader))
    train_correct.append(trn_corr)

    # TEST
    tst_corr = 0
    epoch_test_loss = 0
    model.eval()
    with torch.no_grad():
        for X_test, Y_test in test_loader:
            y_val = model(X_test)

            test_loss = criterion(y_val, Y_test)
            epoch_test_loss += test_loss.item()

            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == Y_test).sum().item()

    test_losses.append(epoch_test_loss / len(test_loader))
    test_correct.append(tst_corr)

    train_acc = (trn_corr / len(train_data)) * 100
    test_acc = (tst_corr / len(test_data)) * 100
    print(f'Epoch {i+1}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.2f}%')

current_time = time.time()
total = current_time - start_time
print(f'\nTraining Time: {total/60:.2f} Minutes!')

# Save the trained model
torch.save(model.state_dict(), 'digits_cnn_model.pth')
print('Model weights saved to digits_cnn_model.pth')

# Final evaluation
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for X_test, Y_test in test_loader:
        y_val = model(X_test)
        test_loss += criterion(y_val, Y_test).item()
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == Y_test).sum().item()

print(f'Final Test Loss: {test_loss/len(test_loader):.4f}')
print(f'Final Test Accuracy: {(correct/len(test_data))*100:.2f}%')
