import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

def save_metrics(all_labels, all_preds, class_names, all_probs, metrics_path="metrics.txt"):
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = (cm.diagonal().sum() / cm.sum()) * 100

    try:
        with open(metrics_path, "w") as f:
            f.write("Raport klasyfikacji:\n")
            f.write(report + "\n")
            f.write(f"Dokładność: {accuracy:.2f}%\n")
        print(f"Metryki zapisano: {metrics_path}")
    except Exception as e:
        print(f"Bład zapisywania metryk: {e}")

    return accuracy, cm


def plot_confusion_matrix(cm, class_names, save_cm_path="confusion_matrix.svg"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Macierz pomyłek")
    ax.set_xlabel("Klasa predykowana")
    ax.set_ylabel("Klasa rzeczywista")
    plt.savefig(save_cm_path, format='svg')
    print(f"Metryki zapisane do: {save_cm_path}")
    plt.show()


def plot_accuracy_loss(history, loss_path="loss_curve.svg", acc_path="accuracy_curve.svg"):
    if history:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(history['train_loss'], label='Funckja straty treningu')
        ax.plot(history['val_loss'], label='Funckja straty walidacji')
        ax.set_xlabel("Epoka")
        ax.set_ylabel("Funkcja straty")
        ax.legend()
        plt.title("Krzywa funkcji straty")
        plt.savefig(loss_path, format="svg")
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(history['train_acc'], label='Dokładność treningu')
        ax.plot(history['val_acc'], label='Dokładność walidacji')
        ax.set_xlabel("Epoka")
        ax.set_ylabel("Funkcja straty")
        ax.legend()
        plt.title("Krzywa dokładności")
        plt.savefig(acc_path, format="svg")
        plt.show()


def plot_roc_auc(all_labels, all_probs, class_names, save_path="roc_auc_curve.svg"):
    """Rysuje i zapisuje wykres ROC-AUC dla każdej klasy."""
    probs = np.asarray(all_probs)
    n_classes = len(class_names)

    plt.figure(figsize=(10, 8))

    if n_classes == 2:
        # Binary classification – probability of positive class (column 1)
        fpr, tpr, _ = roc_curve(all_labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    else:
        # Multi-class classification – One-vs-Rest
        y_true_bin = label_binarize(all_labels, classes=np.arange(n_classes))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Współczynnik wyników fałszywie dodatnich")
    plt.ylabel("Współczynnik wyników prawdziwie pozytywnych")
    plt.title("Krzywa ROC–AUC")
    plt.legend(loc="lower right")
    plt.savefig(save_path, format="svg")
    plt.show()



def plot_prediction_distribution(all_preds, class_names, save_path="prediction_histogram.svg"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(all_preds, bins=len(class_names), edgecolor='black', alpha=0.7)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_xlabel("Predykowana klasa")
    ax.set_ylabel("Czestość wystąpień")
    ax.set_title("Rozkład predykcji")
    plt.savefig(save_path, format="svg")
    plt.show()


def analyze_results(all_labels, all_preds, all_probs, class_names, history=None):
    accuracy, cm = save_metrics(all_labels, all_preds, class_names, all_probs)
    plot_confusion_matrix(cm, class_names)
    plot_roc_auc(all_labels, all_probs, class_names)
    plot_prediction_distribution(all_preds, class_names)
    plot_accuracy_loss(history)

    if history:
        plot_accuracy_loss(history)
