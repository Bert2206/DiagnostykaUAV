import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class UAVClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(UAVClassifier, self).__init__()
        layers = []

        # Warstwa wejściowa
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Ukryte warstwy
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        # Warstwa wyjściowa
        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
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

    # Obliczenie metryk
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Wyświetlenie raportu
    print("Classification Report:\n", report)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(30, 30))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Zapis wykresu, jeśli podano ścieżkę
    if save_cm_path:
        plt.savefig(save_cm_path)
        print(f"Macierz pomyłek została zapisana w pliku: {save_cm_path}")
    else:
        plt.show()

    accuracy = (cm.diagonal().sum() / cm.sum()) * 100
    return accuracy


def save_model(model, scaler, class_names, file_path):
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'class_names': class_names
        }, file_path)
        print(f"Model został zapisany w pliku: {file_path}")
    except Exception as e:
        print(f"Wystąpił błąd podczas zapisywania modelu: {e}")
