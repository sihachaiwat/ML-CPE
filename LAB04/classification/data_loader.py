from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / "data-animal" / "animal_dataset.csv"

def load_data(test_size=0.2, val_size=0.2, random_state=42):

    df = pd.read_csv(DATA_PATH)

    # Target diet type
    y_raw = df['Diet_Type'].values
    
    # Numeric features
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    X_df = numeric_df
        
    X = X_df.values
    feature_names = list(X_df.columns)

    # convert result (target) to numeric
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = [str(c) for c in encoder.classes_]

    # split data เป็น train 60 / validation 20 / test 20
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": class_names,
        "n_rows": len(df),
        "feature_names": feature_names
    }

if __name__ == "__main__":
    data = load_data()
    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("class  :", data["class_names"])