import json
import os
import shutil
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history, plot_batch_comparison

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataImages")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 128
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 1000   # None = use all images
EPOCHS = 100
BATCH_SIZES = [16, 32, 64]


def main():

    print("--" * 30)
    print("CNN Image Recognition Leaf Diseases")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)
    y = labels

    np.save(f"{OUTPUT_DIR}/features.npy", X)

    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # Step 4: Train Model for Each Batch Size
    print("\n[Step 4] Training model with different batch sizes...")

    history_files = []
    plot_labels = []
    last_model = None

    for bs in BATCH_SIZES:
        print("\n" + "=" * 30)
        print(f"   Training Model with Batch Size = {bs}")
        print("=" * 30)

        # เทรนโมเดลโดยระบุ batch_size ตามรอบลูป
        model, history = train_model(
            X_train, y_train, X_val, y_val, len(classes),
            OUTPUT_DIR, EPOCHS, batch_size=bs
        )

        last_model = model

        # ป้องกันไม่ให้ไฟล์ history.json ถูกบันทึกทับในรอบถัดไป
        old_hist_path = os.path.join(OUTPUT_DIR, "history.json")
        new_hist_path = os.path.join(OUTPUT_DIR, f"history_bs_{bs}.json")
        shutil.move(old_hist_path, new_hist_path)

        history_files.append(new_hist_path)
        plot_labels.append(f"Batch Size {bs}")

    print("Training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model (Last Batch Size)...")
    predictions = predict_model(last_model, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    
    comparison_save_path = os.path.join(OUTPUT_DIR, "batch_size_comparison.png")
    plot_batch_comparison(history_files, plot_labels, save_path=comparison_save_path)


if __name__ == "__main__":
    main()