import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class ModifiedBinaryMultiOutputClass:
    def __init__(self, csv_path):
        self.csvPath = csv_path
        with open("hyper.yaml", 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self.batch_size = config['batchSize']
        self.learning_rate = config['learningRate']
        self.epochs = config['epochs']
        self.train_ratio = config['trainRatio']

    def dataload(self):
        df = pd.read_csv(self.csvPath)

        # Replace infs with NaNs and drop any rows with NaN
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Separate cleaned features and target
        X = df.iloc[:, 1:-2].values.astype(np.float32)
        y = df.iloc[:, -2:].values.astype(np.float32)

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create torch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Wrap in a TensorDataset
        full_dataset = TensorDataset(X_tensor, y_tensor)

        # Train/test split
        train_size = int(self.train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle = False)

        return train_loader, test_loader

    def model(self):
        class BinaryMultiOutputModel(nn.Module):
            def __init__(self):
                super(BinaryMultiOutputModel, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(4, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 2),
                    nn.Sigmoid()  # Use Sigmoid for binary outputs
                )

            def forward(self, x):
                return self.net(x)

        return BinaryMultiOutputModel()

    def train(self, model, train_loader, test_loader):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_losses = []
        test_losses = []
        label1_precisions = []
        label2_precisions = []

        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze(-1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            testing_loss, precisions = self.evaluate(model, test_loader, return_metrics=True)
            test_losses.append(testing_loss)
            label1_precisions.append(precisions[0])
            label2_precisions.append(precisions[1])

            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {testing_loss:.4f}")

            if abs(testing_loss - avg_train_loss) > 0.08:
                print("Early stopping triggered")
                break

        self.plot_metrics(train_losses, test_losses, label1_precisions, label2_precisions)

    def evaluate(self, model, dataloader, return_metrics=False):
        model.eval()
        all_outputs = []
        all_targets = []
        total_loss = 0
        criterion = nn.BCELoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                predictions = (outputs >= 0.15).float()
                predictions[:, 1] = torch.where((predictions[:, 0] == 0) & (predictions[:, 1] == 1),
                                                torch.tensor(0.0, device=predictions.device),
                                                predictions[:, 1])

                all_outputs.append(predictions)
                all_targets.append(targets)

        avg_loss = total_loss / len(dataloader)
        all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
        all_targets = torch.cat(all_targets, dim=0).cpu().numpy()

        joint_accuracy = ((all_outputs == all_targets).all(axis=1)).mean()
        print(f"  Joint accuracy       : {joint_accuracy:.4f}")

        precisions = []
        for i in range(2):
            precision = precision_score(all_targets[:, i], all_outputs[:, i], zero_division=0)
            recall = recall_score(all_targets[:, i], all_outputs[:, i], zero_division=0)
            f1 = f1_score(all_targets[:, i], all_outputs[:, i], zero_division=0)
            precisions.append(precision)
            print(f"\nLabel {i + 1}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall   : {recall:.4f}")
            print(f"  F1 Score : {f1:.4f}")

        if return_metrics:
            return avg_loss, precisions
        return avg_loss

    def plot_metrics(self, train_losses, test_losses, label1_precisions, label2_precisions):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 6))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Plot precision
        plt.subplot(1, 2, 2)
        plt.plot(epochs, label1_precisions, label='Precision Label 1')
        plt.plot(epochs, label2_precisions, label='Precision Label 2')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Precision over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    fp = "/Users/mitchelhuott/Downloads/IotaEyewallDataset.csv"
    fp1 = r'/Volumes/MHUOTT_TC/Hurricane Research/Tropical Cylone/CompleteDatasets/GenevieveEyewallDataset.csv'

    user = ModifiedBinaryMultiOutputClass(fp)
    train_loader, test_loader = user.dataload()
    model = user.model()
    user.train(model, train_loader, test_loader)
    user.evaluate(model, test_loader)

    print("GeneveiveEyewallDataset.csv")
    Genevieve = ModifiedBinaryMultiOutputClass(fp1)
    train_loader1, test_loader1 = Genevieve.dataload()
    Genevieve.evaluate(model, test_loader1)
