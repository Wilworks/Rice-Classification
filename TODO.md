# TODO: Fix Rice Classification Notebook Errors

- [x] Fix CSV loading path to 'riceClassification.csv'
- [x] Correct preprocessing: df.dropna(inplace=True), df.drop(['id'], axis=1, inplace=True), fix df[column] assignment
- [x] Fix X and y extraction: X = df.drop("Class", axis=1), y = df["Class"]
- [x] Convert y to numeric labels (e.g., map strings to 0/1)
- [x] Fix dataset class: use .values for torch.tensor, correct dtype typo
- [x] Fix typos in Rice class: nn.Module, nn.Linear, forward indentation
- [x] Correct train/val split: split X_train and y_train properly
- [x] Fix DataLoader variable names: training_Data, etc.
- [x] Initialize plot lists as empty lists []
- [x] Fix training loop: correct variable names, criterion, .item(), etc.
- [x] Fix test section: correct variable names and print statement
- [x] Set device properly for MPS or CPU
