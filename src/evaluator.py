"""
evaluator.py
Evaluates the trained model and generates visualizations:
  - Training loss / accuracy curves
  - Confusion matrix heatmap
  - Classification report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.data_generator import LABEL_MAP

CLASS_NAMES = list(LABEL_MAP.values())   # ["Minimal", "Mild", "Moderate", "Severe"]


def evaluate(model: tf.keras.Model, data: dict) -> None:
    """Print loss, accuracy, and full classification report on the test set."""
    loss, acc = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    print(f"\nTest Loss     : {loss:.4f}")
    print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f}%)\n")

    y_pred = np.argmax(model.predict(data["X_test"], verbose=0), axis=1)
    print("Classification Report:")
    print(classification_report(data["y_test"], y_pred, target_names=CLASS_NAMES))


def plot_training_history(history: tf.keras.callbacks.History, save_path: str = "models/training_curves.png") -> None:
    """Plot and save training vs validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


def plot_confusion_matrix(model: tf.keras.Model, data: dict, save_path: str = "models/confusion_matrix.png") -> None:
    """Plot and save a normalized confusion matrix heatmap."""
    y_pred = np.argmax(model.predict(data["X_test"], verbose=0), axis=1)
    cm = confusion_matrix(data["y_test"], y_pred, normalize="true")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (Normalized)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")
