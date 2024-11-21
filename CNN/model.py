import torch
import os
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class UAVCNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UAVCNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Dodajemy warstwę pool1
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # Dodajemy warstwę pool2
        self.fc1 = nn.Linear(64 * (input_dim // 4), 128)  # Zakładamy 2 warstwy pooling (dzielące przez 4)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # Usunięcie zbędnych wymiarów (jeśli są)
        if x.dim() == 4:  # Jeśli dane mają 4 wymiary
            x = x.squeeze(1)  # Usuń wymiar o rozmiarze 1

        # Przekształcenie na [batch_size, 1, sequence_length]
        x = x.view(x.size(0), 1, -1)
        # Przekazywanie danych przez warstwy CNN
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Spłaszczenie do wektora
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=1):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # Dodanie wymiaru kanału dla CNN
            batch_X = batch_X.unsqueeze(1)
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
    plt.title("Confusion Matrix_CNN")

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
