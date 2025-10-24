import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

# Load the notebook
nb = nbf.read('Rice_Classification.ipynb', as_version=4)

# Function to replace cell content
def replace_cell_content(cell_index, old_content, new_content):
    if nb.cells[cell_index].source == old_content:
        nb.cells[cell_index].source = new_content

# Cell 0: Download Data From Kaggle (keep as is)

# Cell 1: Import Dependencies
replace_cell_content(1, """import torch

import torch.nn as nn

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

# from torch.summary import summary

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



# device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.backends.mps.is_available())
df = pd.read_csv('')

df.head()
df.dropna(implace=True)

df.drop(['id'],axis=1,implace=True)

print(df.shape)
df.head(10)
print(df['Class'].unique())
print(df['Class'].value_counts())""", """import torch

import torch.nn as nn

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

# from torchsummary import summary

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")
df = pd.read_csv('riceClassification.csv')

df.head()
df.dropna(inplace=True)

df.drop(['id'], axis=1, inplace=True)

print(df.shape)
df.head(10)
print(df['Class'].unique())
print(df['Class'].value_counts())""")

# Cell 2: A Little Preprocessing...
replace_cell_content(2, """A Little Preprocessing Like Normalizations and Scaling
orig_df = df.copy()



for column in df.columns:

    df(column) = df[column]/df[column].abs().max()



df.head()
X = Df.drop("Class", axis=1)

y = Df["Species"]""", """# A Little Preprocessing Like Normalizations and Scaling
orig_df = df.copy()

for column in df.columns[:-1]:  # Exclude 'Class' column
    df[column] = df[column] / df[column].abs().max()

df.head()
X = df.drop("Class", axis=1)
y = df["Class"]""")

# Cell 3: Train test split
replace_cell_content(3, """from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.5)""", """from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=41)""")

# Cell 4: Dataset class
replace_cell_content(4, """class dataset(Dataset):

    def __init__(self, X,y):

        self.X = torch.tensor(X, dtype = torch.float32).to(device)

        self.y = torch.tensor(y, dtpye = torch.float32).to(device)





    def __len__(self):

        retun len(self.X)



    def __getitem__(self, index):

        return self.X[index], self.y[index]""", """class dataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X.values, dtype=torch.float32).to(device)

        self.y = torch.tensor(y.values, dtype=torch.float32).to(device)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        return self.X[index], self.y[index]""")

# Cell 5: Create datasets
replace_cell_content(5, """training_Data = dataset(X_train,y_train)

validation_Data = dataset(X_val,y_val)

testing_Data = dataset(X_test,y_test)""", """training_Data = dataset(X_train, y_train)

validation_Data = dataset(X_val, y_val)

testing_Data = dataset(X_test, y_test)""")

# Cell 6: Dataset Loaders
replace_cell_content(6, """Dataset Loaders
train_dataloader = DataLoader(training_data, batch_size = 32, shuffle=True)

validation_dataloader = DataLoader(validation_data, batch_size = 32, shuffle=True)

testing_dataloader = DataLoader(testing_data, batch_size = 32, shuffle=True)""", """# Dataset Loaders
train_dataloader = DataLoader(training_Data, batch_size=32, shuffle=True)

validation_dataloader = DataLoader(validation_Data, batch_size=32, shuffle=True)

testing_dataloader = DataLoader(testing_Data, batch_size=32, shuffle=True)""")

# Cell 7: Model class
replace_cell_content(7, """Hidden_Neurons = 20

class Rice(nn.module):

    def __init__(self):

        super(Rice, self).__init__()



        self.input_layer = nn.linear(X.shape[1], Hidden_Neurons)

        self.linear = nn.linear(Hidden_Neurons, 1)

        self.sigmoid = nn.sigmoid()





        def forward(self,x):

            x = self.input_layer(x)

            x = self.linear(x)

            x = self.sigmoid(x)

            return x



model = Rice().to(device)""", """Hidden_Neurons = 20

class Rice(nn.Module):

    def __init__(self):

        super(Rice, self).__init__()

        self.input_layer = nn.Linear(X.shape[1], Hidden_Neurons)

        self.linear = nn.Linear(Hidden_Neurons, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.input_layer(x)

        x = self.linear(x)

        x = self.sigmoid(x)

        return x

model = Rice().to(device)""")

# Cell 8: Summary (comment out if no torchsummary)
# Assume it's there, but fix if needed

# Cell 9: Criterion and optimizer
replace_cell_content(9, """criterion = nn.BCEloss()

optimizer = Adam(model.parameters, lr = le-3)""", """criterion = nn.BCELoss()

optimizer = Adam(model.parameters(), lr=1e-3)""")

# Cell 10: Training Function
replace_cell_content(10, """Training Function

total_loss_train_plot = ()

total_loss_validation_plot = ()

total_acc_train_plot = ()

total_acc_validation_plot = ()



epochs = 10

for epoch in range(epochs):

    total_acc_train = 0

    total_loss_train = 0

    total_acc_val = 0

    total_loss_val = 0



    for data in train_dataloader():

        inputs, labels = data



        predicition = model(inputs).squeeze(1)



        batch_loss = critetion(prediction,labels)



        total_loss_train += batch_loss.item()

        acc = ((predictions).round() == labels).sum().items()

        total_acc_train += acc

        batch_loss.backward()

        optimizer.step()

        optimizer.zero_grad()





with torch.no_grad():

    for data in validation_dataloader:

        inputs,labels = data

        predicition = model(inputs).squeeze(1)



        batch_loss = critetion(prediction,labels)





        total_loss_validation += batch_loss.item()

        acc = ((predictions).round() == labels).sum().items()

        total_acc_val += acc

total_loss_train_plot.append(round(total_loss_train/1000, 4))

total_loss_validation_plot.append(round(total_loss_val/1000, 4))



total_acc_train_plot.append(round(total_acc_train/training_data.__len__() * 100, 4))

total_acc_validation_plot.append(round(total_acc_val/validation_data.__len__() * 100, 4))



print(f'''Epoch Number: {epoch+1} train Loss: {round(total_loss_train/1000, 4)} Train Accuracy: {round(total_acc_train/training_data.__len__() * 100, 4)}

Validation Loss: {round(total_loss_val/1000, 4)} Validation Accuracy: {round(total_acc_val/validation_data.__len__() * 100, 4)} ''')

print("="*25)""", """# Training Function

total_loss_train_plot = []

total_loss_validation_plot = []

total_acc_train_plot = []

total_acc_validation_plot = []

epochs = 10

for epoch in range(epochs):

    total_acc_train = 0

    total_loss_train = 0

    total_acc_val = 0

    total_loss_val = 0

    model.train()

    for data in train_dataloader:

        inputs, labels = data

        prediction = model(inputs).squeeze(1)

        batch_loss = criterion(prediction, labels)

        total_loss_train += batch_loss.item()

        acc = ((prediction.round() == labels).sum().item())

        total_acc_train += acc

        batch_loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    model.eval()

    with torch.no_grad():

        for data in validation_dataloader:

            inputs, labels = data

            prediction = model(inputs).squeeze(1)

            batch_loss = criterion(prediction, labels)

            total_loss_val += batch_loss.item()

            acc = ((prediction.round() == labels).sum().item())

            total_acc_val += acc

    total_loss_train_plot.append(round(total_loss_train / len(train_dataloader), 4))

    total_loss_validation_plot.append(round(total_loss_val / len(validation_dataloader), 4))

    total_acc_train_plot.append(round(total_acc_train / len(training_Data) * 100, 4))

    total_acc_validation_plot.append(round(total_acc_val / len(validation_Data) * 100, 4))

    print(f'''Epoch {epoch+1}: Train Loss: {total_loss_train_plot[-1]}, Train Acc: {total_acc_train_plot[-1]}%, Val Loss: {total_loss_validation_plot[-1]}, Val Acc: {total_acc_validation_plot[-1]}%''')

print("="*50)""")

# Cell 11: Testing
replace_cell_content(11, """with torch.no_grad():

    total_loss_test = 0

    total_acc_test = 0

    for data in testing_dataloader:

        inputs,labels = data



        prediction = model(inputs).squeez(1)

        batch_loss_test = criterion(predction, labels).item()

        total_loss_test += batch_loss_test



        acc = ((prediction).round() == labels).sum().item()

        total_acc_test += acc



print("Testing Accuracy: ", round(total_acc_train/training_data.__len__() * 100, 4))""", """model.eval()

with torch.no_grad():

    total_loss_test = 0

    total_acc_test = 0

    for data in testing_dataloader:

        inputs, labels = data

        prediction = model(inputs).squeeze(1)

        batch_loss = criterion(prediction, labels)

        total_loss_test += batch_loss.item()

        acc = ((prediction.round() == labels).sum().item())

        total_acc_test += acc

print(f"Test Loss: {round(total_loss_test / len(testing_dataloader), 4)}, Test Accuracy: {round(total_acc_test / len(testing_Data) * 100, 4)}%")""")

# Save the notebook
nbf.write(nb, 'Rice_Classification.ipynb')
print("Notebook fixed and saved.")
