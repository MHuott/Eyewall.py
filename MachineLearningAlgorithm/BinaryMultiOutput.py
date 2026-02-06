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
import os

class BinaryMultiOutputClass:
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
                    #nn.Linear(4, 32),
                   # nn.ReLU(),
                    nn.Linear(4, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 4),
                    nn.ReLU(),
                    nn.Linear(4, 2),
                    nn.Sigmoid()  # Use Sigmoid for binary outputs
                )

            def forward(self, x):
                return self.net(x)

        return BinaryMultiOutputModel()

    def train(self, model, train_loader, test_loader):
        criterion = nn.BCELoss()  # Binary Cross-Entropy for sigmoid outputs
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_losses = []
        test_losses = []
        label1_precisions = []
        label2_precisions = []
        joint_accuracies = []
        epoch_summaries = []

        for epoch in range(self.epochs):
            model.train()
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

            # Evaluate on test set to compute precision
            model.eval()
            all_preds, all_targets = [], []

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            testing_loss, precisions, joint_acc = self.evaluate(model, test_loader, return_metrics=True)

            test_losses.append(testing_loss)
            label1_precisions.append(precisions[0])
            label2_precisions.append(precisions[1])
            joint_accuracies.append(joint_acc)

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    preds = (outputs >= 0.5).float()
                    all_preds.append(preds)
                    all_targets.append(targets)

            all_preds = torch.cat(all_preds).cpu().numpy()
            all_targets = torch.cat(all_targets).cpu().numpy()

            # Compute average precision over both labels
            precision = np.mean([
                precision_score(all_targets[:, i], all_preds[:, i], zero_division=0)
                for i in range(2)
            ])

            epoch_summaries.append({
                "epoch": epoch + 1,
                "train_loss": total_loss,
                "test_loss": testing_loss,
                "precision_label1": precisions[0],
                "precision_label2": precisions[1],
                "joint_accuracy": joint_acc
            })

        # Plotting
        self.plot_metrics(train_losses, test_losses, label1_precisions, label2_precisions, joint_accuracies)

        print("\n📊 Training Summary:")
        for summary in epoch_summaries:
            print(
                f"Epoch {summary['epoch']:2d} | "
                f"Train Loss: {summary['train_loss']:.4f} | "
                f"Test Loss: {summary['test_loss']:.4f} | "
                f"Precision L1: {summary['precision_label1']:.4f} | "
                f"Precision L2: {summary['precision_label2']:.4f} | "
                f"Joint Acc: {summary['joint_accuracy']:.4f}"
            )

    def evaluate(self, model, dataloader, return_metrics=False, verbose=False):
        model.eval()
        all_outputs = []
        all_targets = []
        total_loss = 0
        criterion = nn.BCELoss()

        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                predictions = (outputs >= 0.15).float()
                all_outputs.append(predictions)
                all_targets.append(targets)

        all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
        all_targets = torch.cat(all_targets, dim=0).cpu().numpy()

        label_accuracies = (all_outputs == all_targets).mean(axis=0)
        joint_accuracy = ((all_outputs == all_targets).all(axis=1)).mean()

        label_precisions = [
            precision_score(all_targets[:, i], all_outputs[:, i], zero_division=0)
            for i in range(2)
        ]

        if verbose:
            print("🔎 Evaluation:")
            print(f"  Accuracy for label 1: {label_accuracies[0]:.4f}")
            print(f"  Accuracy for label 2: {label_accuracies[1]:.4f}")
            print(f"  Joint accuracy       : {joint_accuracy:.4f}")
            for i in range(2):
                recall = recall_score(all_targets[:, i], all_outputs[:, i], zero_division=0)
                f1 = f1_score(all_targets[:, i], all_outputs[:, i], zero_division=0)
                print(f"\nLabel {i + 1}:")
                print(f"  Precision: {label_precisions[i]:.4f}")
                print(f"  Recall   : {recall:.4f}")
                print(f"  F1 Score : {f1:.4f}")

        if return_metrics:
            avg_loss = total_loss / len(dataloader)
            return avg_loss, label_precisions, joint_accuracy

    def predict_and_append_labels(self, model, new_csv_path, output_path=None, threshold=0.5):
        """
        Predicts eyewall labels on a new dataset using a trained model and appends predictions.

        Args:
            model (torch.nn.Module): Trained binary multi-output model.
            new_csv_path (str): Path to the new CSV to label.
            output_path (str): Optional path to save the output CSV with labels.
            threshold (float): Threshold for converting probabilities to binary predictions.

        Returns:
            pd.DataFrame: DataFrame with added 'PrimaryEyewall' and 'SecondaryEyewall' columns.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)

        # Load and clean the new CSV
        df = pd.read_csv(new_csv_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Extract features (assumes same format as training data)
        X_new = df.iloc[:, 1:-2].values.astype(np.float32)

        # Apply the same scaling
        scaler = StandardScaler()
        X_new_scaled = scaler.fit_transform(X_new)  # Use same scaling logic as training; ideally use saved scaler

        # Convert to tensor
        X_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

        # Run model predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = (outputs >= threshold).float().cpu().numpy()

        # Append predictions to original DataFrame
        df['PrimaryEyewall'] = predictions[:, 0].astype(int)
        df['SecondaryEyewall'] = predictions[:, 1].astype(int)

        # Optionally save the output
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✅ Predictions saved to {os.path.abspath(output_path)}")

        return df

    def plot_metrics(self, train_losses, test_losses, label1_precisions, label2_precisions, joint_accuracies):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(18, 5))

        # Plot losses
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Plot precision
        plt.subplot(1, 3, 2)
        plt.plot(epochs, label1_precisions, label='Precision Label 1')
        plt.plot(epochs, label2_precisions, label='Precision Label 2')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Precision over Epochs')
        plt.legend()

        # Plot joint accuracy
        plt.subplot(1, 3, 3)
        plt.plot(epochs, joint_accuracies, label='Joint Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Joint Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig("independent.png")
        plt.show()

    def save_model(self, model, path="model.pth"):
        """Save the model parameters and training hyper‑params."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'train_ratio': self.train_ratio,
        }, path)
        print(f"✅ Model saved to {os.path.abspath(path)}")

    def load_model(self, model, path="model.pth", map_location=None):
        """Load parameters into an *already‑constructed* model object."""
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📂 Loaded weights from {os.path.abspath(path)}")
        return model

if __name__ == "__main__":
    fp1 = r'/Volumes/MHUOTT_PHYS/Hurricane Research/Tropical Cylone/CompleteDatasets/Smoothed Genevieve Eyewall.csv'
    fp2 = r'/Volumes/MHUOTT_PHYS/Hurricane Research/Tropical Cylone/CompleteDatasets/Smoothed Larry Eyewall2.csv'
    fp3 = r'/Volumes/MHUOTT_PHYS/Hurricane Research/Tropical Cylone/CompleteDatasets/Smoothed Sam Eyewall4.csv'
    fp4 = r'/Volumes/MHUOTT_PHYS/Hurricane Research/Tropical Cylone/Datasets In-Progress/Linda Eyewall.csv'

    Genevieve = BinaryMultiOutputClass(fp1)
    train_loader1, test_loader1 = Genevieve.dataload()

    Larry = BinaryMultiOutputClass(fp2)
    train_loader2, test_loader2 = Larry.dataload()

    Sam = BinaryMultiOutputClass(fp3)
    train_loader3, test_loader3 = Sam.dataload()

    model = Genevieve.model()

    #Sam.train(model, train_loader3, test_loader3)
    #Larry.train(model, train_loader2, test_loader2)
    #Sam.train(model, train_loader3, test_loader3)
    #Genevieve.train(model, train_loader1, test_loader1)
    #Sam.train(model, train_loader3, test_loader3)
    Larry.train(model, train_loader2, test_loader2)
    #Sam.train(model, train_loader3, test_loader3)

    print("Genevieve")
    Genevieve.evaluate(model, test_loader1, verbose=True)

    print("Larry")
    Larry.evaluate(model, test_loader2, verbose=True)

    print("Sam")
    Sam.evaluate(model, test_loader3, verbose=True)

    '''Genevieve.load_model(model, fp4, map_location="cpu")

    # Predict on new data and save to file
    new_data_fp = "/Users/mitchelhuott/Downloads/NewStormEyewall.csv"
    output_fp = "/Users/mitchelhuott/Downloads/NewStorm_with_predictions.csv"
    df_with_preds = Genevieve.predict_and_append_labels(model, new_data_fp, output_path=output_fp)



user.evaluate(model, test_loader)
from your_module import BinaryMultiOutputClass   # wherever the class lives

csv_fp = "/Users/mitchelhuott/Downloads/IotaEyewallDataset.csv"
user    = BinaryMultiOutputClass(csv_fp)

model   = user.model()                 # construct architecture
model   = user.load_model(model, "iota_binary_multilabel.pth",
                          map_location="cpu")  # or "cuda"
model.eval()

# now evaluate or continue training
_, test_loader = user.dataload()
user.evaluate(model, test_loader, verbose=True)'''

