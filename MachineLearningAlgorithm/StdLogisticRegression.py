import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score


csv_path = '/Volumes/MHUOTT_PHYS/Hurricane Research/Tropical Cylone/Combined.csv'
with open("hyper.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
batch_size = config['batchSize']
learning_rate = config['learningRate']
epochs = config['epochs']
train_ratio = config['trainRatio']

# === Load and preprocess data ===
df = pd.read_csv(csv_path)

# Replace infs with NaNs and drop any rows with NaN
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Separate cleaned features and target
X = df.iloc[:, 1:-2].values.astype(np.float32)
y = df.iloc[:, -2:].values.astype(np.float32)


#print(y)
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create torch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)  # shape: (N, 1)

# Wrap in a TensorDataset
full_dataset = TensorDataset(X_tensor, y_tensor)

# Train/test split
train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
# === Logistic Regression Model ===
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_input_features, 2)

    def forward(self, x):
        return self.linear(x)  # logits, no sigmoid here
'''

class LogisticRegressionModel(torch.nn.Module):
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, 2)
    # make predictions
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Initialize model
n_features = X.shape[1]
model = LogisticRegressionModel(n_features, 2)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# === Training Loop ===
for epoch in range(epochs):
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)  # raw logits
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            #print("Found NaN loss! Debug info:")
            #print("Inputs min/max:", inputs.min().item(), inputs.max().item())
            #print("Outputs (logits) min/max:", outputs.min().item(), outputs.max().item())
            #print("Labels min/max:", labels.min().item(), labels.max().item())
            #print("Model weights:", list(model.parameters()))
            break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# === Evaluation ===
with torch.no_grad():
    all_labels = []
    all_preds = []

    for inputs, labels in test_loader:
        logits = model(inputs)
        probs = torch.sigmoid(logits)
        predicted = (probs > 0.05).float()

        all_labels.append(labels)
        all_preds.append(predicted)

    # Concatenate all batches
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()

    # Per-label metrics
    for i in range(all_labels.shape[1]):
        precision = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        recall = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        accuracy = (all_labels[:, i] == all_preds[:, i]).mean()

        print(f"\nLabel {i+1}:")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f1:.4f}")

    # Optional: Joint accuracy (both labels correct at once)
    joint_acc = (all_labels == all_preds).all(axis=1).mean()
    print(f"\nJoint accuracy: {joint_acc:.4f}")
