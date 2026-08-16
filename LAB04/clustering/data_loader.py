## Read data.csv and prepare for clustering

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data-animal" / "animal_dataset.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Numeric features
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    
    features = list(numeric_df.columns)
    X_raw = numeric_df.values

    # Scaling 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return {"X": X_scaled,"X_raw": X_raw,"df": df,"features": features}

if __name__ == "__main__":
    data = load_data()
    print("Features :", data["features"])
    print("size data :", data["X"].shape)
    print("mean after scale (should be close to 0) :", data["X"].mean(axis=0).round(3))