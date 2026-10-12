import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset

from vgg_model import train_model, predict_model, VGG16, VGG_SMALL, VGG_MICRO
from evaluate import evaluate_model, plot_history, plot_model_comparison

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataImages")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 224
TEST_SIZE = 0.2
VAL_SIZE = 0.2
MAX_PER_CLASS = 1500
EPOCHS = 100
BATCH_SIZE = 16

def main():
    print("--" * 30)
    print("VGG (Deep CNN) Image Recognition Leaf Diseases")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1-3: Load, Preprocess, Split 
    print("\n[Step 1-3] Loading and Preparing dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    X = to_features(images)
    y = labels
    np.save(f"{OUTPUT_DIR}/features.npy", X)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, TEST_SIZE, VAL_SIZE)

    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    
    models_to_test = {
        "Model_1_VGG16": VGG16,
        "Model_2_Small": VGG_SMALL,
        "Model_3_Micro": VGG_MICRO
    }
    
    histories = {}
    best_accuracy = 0
    best_model_name = ""

    # Step 4-6: Train, Predict, Evaluate 
    for model_name, blocks in models_to_test.items():
        print(f"\n>>> Training {model_name} <<<")
        model, history = train_model(
            X_train, y_train, X_val, y_val, len(classes),
            output_dir=f"{OUTPUT_DIR}/{model_name}", epochs=EPOCHS, batch_size=BATCH_SIZE, blocks=blocks
        )
        histories[model_name] = history
        plot_history(history, f"{OUTPUT_DIR}/{model_name}_history.png")
        predictions = predict_model(model, X_test)
        print(f"\nEvaluating {model_name}...")
        acc = evaluate_model(y_test, predictions, classes, save_path=f"{OUTPUT_DIR}/{model_name}_cm.png")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = model_name
            # save best model
            model.save(os.path.join(OUTPUT_DIR, "best_model.keras"))

    print("\n" + "="*40)
    print(f"WINNER: {best_model_name} (Accuracy: {best_accuracy * 100:.2f}%)")
    print("="*40)

    plot_model_comparison(histories, f"{OUTPUT_DIR}/comparison_graph.png")

if __name__ == "__main__":
    main()