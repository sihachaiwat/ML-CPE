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
│   ├── cnn_model.py                    # CNN model architecture and training module
│   ├── data_loader.py                  # Dataset loading module
│   ├── evaluate.py                     # Evaluation metrics and plotting utilities
│   ├── main.py                         # Main pipeline execution script
│   ├── plot_all_histories.py           # Script to generate individual training graphs
│   ├── preprocessing.py                # Data preprocessing and feature conversion
│   ├── split_data.py                   # Train/Val/Test dataset splitting module
│   └── test_cnn.py                     # Model inference testing script
├── dataImages/                         # Raw image dataset directory
│   ├── Diseased/                       # Images of diseased plant leaves
│   └── Healthy/                        # Images of healthy plant leaves
├── requirements.txt
├── README.md
└── link-data.txt
```