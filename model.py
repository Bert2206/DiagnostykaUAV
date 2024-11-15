import torch
import os
import torch.nn as nn
import torch.optim as optim

class UAVClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UAVClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def save_model(model, scaler, class_names, file_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'class_names': class_names
    }, file_path)
    if os.path.exists(file_path):
        print(f"Model został zapisany poprawnie w pliku: {file_path}")
    else:
        print(f"Błąd: Nie udało się zapisać pliku w ścieżce: {file_path}")

#except Exception as e:\
#    print(f"Wystąpił błąd podczas zapisywania modelu: {e}")
