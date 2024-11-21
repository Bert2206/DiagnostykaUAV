import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import re
import torch.nn as nn
from model import UAVLSTMClassifier, train_model, evaluate_model, save_model

# Load and preprocess data
folder_path = r'C:\Users\Admin\PycharmProjects\DiagnostykaUAV\Parrot_Bebop_2\Range_data'
file_pattern = folder_path + '\\*.csv'
data_frames = []

for file_path in glob.glob(file_pattern):
    data = pd.read_csv(file_path)
    match = re.search(r'_(\d{4}).csv$', file_path)  # Match fault code
    if match:
        fault_code = match.group(1)
        data['Fault_Code'] = fault_code
    else:
        print(f"Invalid file name: {file_path}")
    data_frames.append(data)

all_data = pd.concat(data_frames, ignore_index=True)

# Encode fault codes
le = LabelEncoder()
y = le.fit_transform(all_data['Fault_Code'])
y = torch.tensor(y, dtype=torch.long)
class_names = [str(cls) for cls in le.classes_]

# Prepare input data
X = all_data.drop(columns=['Fault_Code']).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for LSTM (batch, sequence, features)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Sequence length = 1
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

# Create datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize LSTM model
input_dim = X_train.shape[2]  # Number of features
output_dim = len(torch.unique(y))
model = UAVLSTMClassifier(input_dim=input_dim, output_dim=output_dim)

# Loss, optimizer, and device
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Train model
train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

# Evaluate model
accuracy = evaluate_model(model, test_loader, device, class_names, save_cm_path="confusion_matrix_lstm.png")
print(f"Accuracy: {accuracy:.2f}%")

# Save model
save_model(model, scaler, class_names, "uav_lstm_model.pth")
