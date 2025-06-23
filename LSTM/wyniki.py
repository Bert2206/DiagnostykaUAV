import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    cohen_kappa_score, matthews_corrcoef, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

def save_metrics(all_labels, all_preds, class_names, all_probs, metrics_path="metrics.txt"):
    """Zapisuje metryki klasyfikacji do pliku."""
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = (cm.diagonal().sum() / cm.sum()) * 100

    try:
        with open(metrics_path, "w") as f:
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")
        print(f"Metrics saved to: {metrics_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    return accuracy, cm


def plot_confusion_matrix(cm, class_names, save_cm_path="confusion_matrix.svg"):
    """Rysuje i zapisuje macierz pomyłek."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig(save_cm_path, format='svg')
    print(f"Confusion matrix saved to: {save_cm_path}")
    plt.show()


def plot_accuracy_loss(history, loss_path="loss_curve.svg", acc_path="accuracy_curve.svg"):
    """Rysuje i zapisuje wykresy loss i accuracy z historii treningu."""
    if history:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(history['train_loss'], label='Train Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.title("Loss Curve")
        plt.savefig(loss_path, format="svg")
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(history['train_acc'], label='Train Accuracy')
        ax.plot(history['val_acc'], label='Validation Accuracy')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        plt.title("Accuracy Curve")
        plt.savefig(acc_path, format="svg")
        plt.show()


def plot_roc_auc(all_labels, all_probs, class_names, save_path="roc_auc_curve.svg"):
    probs = np.asarray(all_probs)
    n_classes = len(class_names)

    plt.figure(figsize=(10, 8))

    if nclasses == 2:
        # Binary classification – take the probability of the positive class (column 1)
        fpr, tpr,  = roc_curve(all_labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    else:
        # Multi-class (one-vs-rest) ROC curves
        y_true_bin = label_binarize(all_labels, classes=np.arange(n_classes))
        for i in range(nclasses):
            fpr, tpr,  = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0]);  plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate");  plt.ylabel("True Positive Rate")
    plt.title("ROC–AUC Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_path, format="svg")
    plt.show()


def plot_prediction_distribution(all_preds, class_names, save_path="prediction_histogram.svg"):
    """Rysuje histogram przewidywań modelu."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(all_preds, bins=len(class_names), edgecolor='black', alpha=0.7)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Frequency")
    ax.set_title("Prediction Distribution")
    plt.savefig(save_path, format="svg")
    plt.show()


def analyze_results(all_labels, all_preds, all_probs, class_names, history=None):
    """Wykonuje pełną analizę wyników modelu."""
    accuracy, cm = save_metrics(all_labels, all_preds, class_names, all_probs)
    plot_confusion_matrix(cm, class_names)
    plot_roc_auc(all_labels, all_probs, class_names)
    plot_prediction_distribution(all_preds, class_names)
    plot_accuracy_loss(history)

    if history:
        plot_accuracy_loss(history)
