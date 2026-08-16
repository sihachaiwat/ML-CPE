from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_k_curve(k_values, scores, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker='o', linestyle='-', color='b')
    plt.title('Validation Accuracy vs. K Value')
    plt.xlabel('K Value')
    plt.ylabel('Validation Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_report(y_true, y_pred, class_names):
    str_class_names = [str(c) for c in class_names]
    print(classification_report(y_true, y_pred, target_names=str_class_names))

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    str_class_names = [str(c) for c in class_names]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=str_class_names, yticklabels=str_class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return cm

def save_predictions(y_true, y_pred, class_names, save_path):
    df = pd.DataFrame({
        'True_Label': [class_names[i] for i in y_true],
        'Predicted_Label': [class_names[i] for i in y_pred],
        'Correct': y_true == y_pred
    })
    df.to_csv(save_path, index=False)