import matplotlib

import json
import matplotlib.ticker as mtick

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)
    print("Confusion Matrix:")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path):
    """Accuracy and loss curves — the main tool for spotting overfitting."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_batch_comparison(history_files, labels, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#00B050', '#FF0000', '#00FFFF', '#FFC000']
    
    for i, (hist_file, label) in enumerate(zip(history_files, labels)):
        with open(hist_file, 'r') as f:
            history_data = json.load(f)
        
        val_acc_pct = [acc * 100 for acc in history_data['val_accuracy']]
        ax.plot(
            range(1, len(val_acc_pct) + 1), 
            val_acc_pct, 
            label=label, 
            color=colors[i % len(colors)], 
            linewidth=2
        )

    ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim([0, 100])
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.15), 
        ncol=len(labels), 
        edgecolor='black'
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")