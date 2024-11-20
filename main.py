import pandas as pd
import re
import glob
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from model import UAVClassifier, train_model, evaluate_model, save_model

# Wczytanie danych i przemapowanie kodów błędów
folder_path = r'C:\Users\Admin\PycharmProjects\DiagnostykaUAV\Parrot_Bebop_2\Range_data'
file_pattern = folder_path + '\\*.csv'
data_frames = []

for file_path in glob.glob(file_pattern):
    data = pd.read_csv(file_path)
    match = re.search(r'_(\d{4}).csv$', file_path)  # Dopasowanie dokładnie czterech cyfr
    if match:
        fault_code = match.group(1)  # Zachowujemy fault_code jako tekst
        data['Fault_Code'] = fault_code
    else:
        print(f"Nieprawidłowa nazwa pliku: {file_path}")
    data_frames.append(data)

all_data = pd.concat(data_frames, ignore_index=True)

# Przekształcanie kodów błędów na indeksy klas
le = LabelEncoder()
y = le.fit_transform(all_data['Fault_Code'])
y = torch.tensor(y, dtype=torch.long)
class_names = [str(cls) for cls in le.classes_]  # Lista nazw klas

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

# Inicjalizacja modelu, funkcji straty i optymalizatora
input_dim = X_train.shape[1]
output_dim = len(torch.unique(y))
# Definiowanie liczby i rozmiaru warstw ukrytych
hidden_layers = [128, 64, 32]  # Trzy warstwy ukryte o różnej liczbie neuronów
# Inicjalizacja modelu z nową konfiguracją
model = UAVClassifier(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers)
#model = UAVClassifier(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Trening modelu
train_model(model, train_loader, criterion, optimizer, device)

# Ewaluacja modelu
accuracy = evaluate_model(model, test_loader, device, class_names, save_cm_path="confusion_matrix.png")
print(f"Accuracy: {accuracy:.2f}%")

# Zapis modelu
save_model(model, scaler, class_names, "uav_model.pth")
