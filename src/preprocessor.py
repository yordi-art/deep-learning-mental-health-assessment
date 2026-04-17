"""
preprocessor.py
Handles loading, validation, scaling, and splitting of the mental health dataset.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_generator import ALL_FEATURES, LABEL_MAP


def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset and return feature matrix X and label vector y."""
    df = pd.read_csv(csv_path)

    missing = [c for c in ALL_FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[ALL_FEATURES].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    return X, y


def preprocess(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scaler_path: str = "models/scaler.pkl",
) -> dict:
    """
    Split into train/val/test sets and apply StandardScaler.
    Saves the fitted scaler to disk for use during inference.

    Returns a dict with keys: X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Split remaining into train + validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved → {scaler_path}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
    }


def scale_input(raw_input: np.ndarray, scaler_path: str = "models/scaler.pkl") -> np.ndarray:
    """Load saved scaler and transform a single inference input."""
    scaler = joblib.load(scaler_path)
    return scaler.transform(raw_input.reshape(1, -1))
