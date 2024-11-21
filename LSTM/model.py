import torch
import os
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=1):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


def evaluate_model(model, data_loader, device, class_names, save_cm_path=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    # Display results
    print("Classification Report:\n", report)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(30, 30))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix_LSTM")

    # Save confusion matrix as image, if specified
    if save_cm_path:
        plt.savefig(save_cm_path)
        print(f"Confusion matrix saved to: {save_cm_path}")
    else:
        plt.show()

    # Calculate accuracy
    accuracy = (cm.diagonal().sum() / cm.sum()) * 100
    return accuracy


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
