import pandas as pd
import re
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Wczytanie danych i przemapowanie kodów błędów
folder_path = r'C:\Users\kubas\PycharmProjects\DiagnostykaUAV\Parrot_Bebop_2\Range_data'
file_pattern = folder_path + '\\*.csv'
data_frames = []

for file_path in glob.glob(file_pattern):
    data = pd.read_csv(file_path)
    match = re.search(r'_(\d{4}).csv$', file_path)
    if match:
        fault_code = match.group(1)
        data['Fault_Code'] = int(fault_code)
    else:
        print(f"Nieprawidłowa nazwa pliku: {file_path}")
    data_frames.append(data)

all_data = pd.concat(data_frames, ignore_index=True)

# Przekształcanie kodów błędów na indeksy klas
le = LabelEncoder()
y = le.fit_transform(all_data['Fault_Code'])
y = torch.tensor(y, dtype=torch.long)

# Przygotowanie zmiennych X i y
X = all_data.drop(columns=['Fault_Code']).values

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja danych wejściowych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Konwersja do tensorów
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Tworzenie zbiorów danych i loaderów
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definicja modelu
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

# Inicjalizacja modelu, funkcji straty i optymalizatora
input_dim = X_train.shape[1]
output_dim = len(torch.unique(y))  # Liczba unikalnych klas
model = UAVClassifier(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening modelu
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()  # Ustawienie modelu w tryb ewaluacji
correct = 0
total = 0
with torch.no_grad():  # Wyłączenie gradientów dla ewaluacji
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
