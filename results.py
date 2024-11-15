import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import UAVClassifier


def load_model(model_path, input_dim, output_dim):
    checkpoint = torch.load(model_path)
    model = UAVClassifier(input_dim, output_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']
    class_names = checkpoint['class_names']
    print(f"Model załadowany z {model_path}")
    return model, scaler, class_names


def evaluate_and_display_results(model, test_loader, device, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Obliczanie metryk
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Raport klasyfikacji:")
    print(report)

    # Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predykcja")
    plt.ylabel("Prawdziwa klasa")
    plt.title("Macierz pomyłek")
    plt.show()


if __name__ == "__main__":
    from main import X_test, y_test, train_test_split, DataLoader, TensorDataset

    # Załaduj model i dane
    model_path = "uav_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Informacje o wymiarach danych
    input_dim = X_test.shape[1]
    output_dim = len(torch.unique(y_test))

    model, scaler, class_names = load_model(model_path, input_dim, output_dim)
    model.to(device)

    # Tworzenie test loadera
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Ewaluacja i wizualizacja
    evaluate_and_display_results(model, test_loader, device, class_names)
