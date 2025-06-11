import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.model_selection import train_test_split  # for train/test split

class CSVDataset(Dataset):
    def __init__(self, csvFile, header = 0):
        # Load the CSV file into a pandas DataFrame
        self.annotations = pd.read_csv(csvFile)

        # Convert all but last two columns to features (X), and last column to labels (y)

    def __len__(self):
        return len(self.y) #length of the entire dataset

    def __getitem__(self, index):
        self.X = torch.tensor(data.iloc[:, 1:-4].values, dtype=torch.float32)
        self.y = torch.tensor(data.iloc[:, -4:-2].values, dtype=torch.float32)
        return self.X[index], self.y[index]

# Create model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


csvPath = r"C:\Users\mlhuo\PycharmProjects\Eyewall.py\netcdf4Images\Eta Eyewall.csv" # Replace with your actual path
dataset = CSVDataset(csvPath)
batchSize = 64

trainDataloader = DataLoader(dataset, batch_size = batchSize, shuffle=True)
testDataloader = DataLoader(dataset, batch_size = (int(batchSize/2)), shuffle=True)




'''
print("training data")
for batch_idx, (features, labels) in enumerate(trainDataloader):
    print(f"Batch {batch_idx+1}")
    print("Features:", features)
    print("Labels:", labels)
    break

print("testing data")
for batch_idx, (features, labels) in enumerate(testDataloader):
    print(f"Batch {batch_idx+1}")
    print("Features:", features)
    print("Labels:", labels)
    break
'''