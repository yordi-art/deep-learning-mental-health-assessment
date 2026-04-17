"""
train.py
Entry point — runs the full training pipeline:
  1. Generate (or load) dataset
  2. Preprocess and split
  3. Build ANN model
  4. Train with callbacks
  5. Evaluate and save visualizations
"""

import os
from src.data_generator import generate_dataset
from src.preprocessor import load_data, preprocess
from src.model import build_model
from src.trainer import train
from src.evaluator import evaluate, plot_training_history, plot_confusion_matrix

DATASET_PATH = "data/mental_health_dataset.csv"
MODEL_PATH   = "models/mental_health_model.keras"
SCALER_PATH  = "models/scaler.pkl"


def main():
    # ── 1. Dataset ──────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        print("Generating synthetic dataset...")
        df = generate_dataset(n_samples=2000)
        df.to_csv(DATASET_PATH, index=False)
        print(f"Dataset saved → {DATASET_PATH}  shape={df.shape}\n")
    else:
        print(f"Dataset found → {DATASET_PATH}\n")

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    X, y = load_data(DATASET_PATH)
    data = preprocess(X, y, scaler_path=SCALER_PATH)
    print(f"Train: {data['X_train'].shape}  Val: {data['X_val'].shape}  Test: {data['X_test'].shape}\n")

    # ── 3. Build model ───────────────────────────────────────────────────────
    model = build_model(input_dim=X.shape[1])
    model.summary()

    # ── 4. Train ─────────────────────────────────────────────────────────────
    history = train(model, data, model_path=MODEL_PATH)

    # ── 5. Evaluate & visualize ──────────────────────────────────────────────
    evaluate(model, data)
    plot_training_history(history)
    plot_confusion_matrix(model, data)

    print("\nPipeline complete. Run `python -m src.predictor` to test inference.")


if __name__ == "__main__":
    main()
