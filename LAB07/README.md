# LAB07 Convolutional Neural Network (CNN)
An end-to-end Deep Learning pipeline built with TensorFlow/Keras to classify plant leaf images into **Diseased** or **Healthy** categories. This project includes modular Python components for image preprocessing, data splitting, model training with regularization, hyperparameter testing (Batch Size comparison), and comprehensive visualization.  

## Dataset
* **Dataset:** Healthy vs Diseased (dataImages) select only apple leaf
* **Link:** https://github.com/spMohanty/PlantVillage-Dataset/tree/master

## Structure
```text
LAB07/
├── classification/                     # Main source code and execution directory
│   ├── outputs/                        # Generated model outputs and evaluation artifacts
│   │   ├── batch_size_comparison.png   # Combined accuracy comparison chart across batch sizes
│   │   ├── classes.json                # Class names mapping JSON
│   │   ├── cnn_model.keras             # Saved trained Keras model file
│   │   ├── confusion_matrix.png        # Confusion matrix visualization
│   │   ├── features.npy                # Extracted image features array
│   │   ├── history_bs_16.json          # Training history for batch size 16
│   │   ├── history_bs_32.json          # Training history for batch size 32
│   │   ├── history_bs_64.json          # Training history for batch size 64
│   │   ├── labels.npy                  # Target labels array
│   │   ├── prediction_sample.png       # Sample predictions visualization
│   │   ├── training_history_bs_16.png  # Accuracy & Loss plot for batch size 16
│   │   ├── training_history_bs_32.png  # Accuracy & Loss plot for batch size 32
│   │   ├── training_history_bs_64.png  # Accuracy & Loss plot for batch size 64
│   │   ├── X_test.npy                  # Test set features
│   │   ├── X_train.npy                 # Training set features
│   │   ├── X_val.npy                   # Validation set features
│   │   ├── y_test.npy                  # Test set labels
│   │   ├── y_train.npy                 # Training set labels
│   │   └── y_val.npy                   # Validation set labels
│   │
│   ├── cnn_model.py                    # CNN model architecture and training module
│   ├── data_loader.py                  # Dataset loading module
│   ├── evaluate.py                     # Evaluation metrics and plotting utilities
│   ├── main.py                         # Main pipeline execution script
│   ├── plot_all_histories.py           # Script to generate individual training graphs
│   ├── preprocessing.py                # Data preprocessing and feature conversion
│   ├── split_data.py                   # Train/Val/Test dataset splitting module
│   └── test_cnn.py                     # Model inference testing script
│
├── dataImages/                         # Raw image dataset directory
│   ├── Diseased/                       # Images of diseased plant leaves
│   └── Healthy/                        # Images of healthy plant leaves
│
├── requirements.txt
├── README.md
└── link-data.txt
```

## Topic
- **Custom CNN Architecture**: Designed with Data Augmentation (RandomRotation, RandomZoom, RandomContrast), Dropout, and EarlyStopping to effectively prevent overfitting.

- **Batch Size Comparison**: Automated experiment pipeline to train and compare performance across multiple batch sizes (16, 32, 64).

- **Comprehensive Visualization**: Automatically generates Top-1 Accuracy comparison graphs, per-batch Loss/Accuracy curves, and Confusion Matrices.

- **Modular Pipeline Structure**: Clean code separation across dataset loading, preprocessing, splitting, training, evaluation, and plotting modules.

## Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the CNN Model:**
   ```bash
   python classification/main.py
   ```

3. **Test & Visualize Predictions:**
   ```bash
   python classification/test_cnn.py
   ```

4. **Plot Individual Training Histories:**
   ```bash
   python classification/plot_all_histories.py
   ```

## Summarize
- Overall Test Accuracy: 95.75% (383 / 400 test images correctly classified)
- Batch Size Performance Analysis:
    - Batch Size 16: Fast early convergence, but exhibits higher gradient variance.
    - Batch Size 32 (Optimal): Best balance between learning speed, gradient stability, and final accuracy.
    - Batch Size 64: Smooth loss reduction, but requires more epochs to converge.