import torch
import os
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from wyniki import analyze_results


class UAVLSTMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, bidirectional=False, dropout=0.2):
        super(UAVLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0  # Dropout only applies when num_layers > 1
        )

        # Fully Connected Layer for Classification
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)

    def forward(self, x):
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Use the last time-step's output for classification
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=1, test_loader=None):
    model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = (correct_train / total_train) * 100
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Ewaluacja na zbiorze walidacyjnym (opcjonalna)
        if test_loader:
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            val_loss = running_val_loss / len(test_loader)
            val_acc = (correct_val / total_val) * 100
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%"
                  f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        else:
            print(f"Epoch [{epoch}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

    return history



def evaluate_model(model, data_loader, device, class_names, history=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Przekazanie wyników do `wyniki.py`
    analyze_results(all_labels, all_preds, all_probs, class_names, history)

def save_model(model, scaler, class_names, file_path):
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'class_names': class_names
        }, file_path)
        if os.path.exists(file_path):
            print(f"Model successfully saved to: {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
