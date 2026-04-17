"""
predictor.py
Loads the saved model and scaler to predict mental health severity
from new PHQ-9 + GAD-7 questionnaire responses.
"""

import numpy as np
import tensorflow as tf

from src.data_generator import LABEL_MAP, PHQ9_COLS, GAD7_COLS
from src.preprocessor import scale_input


def load_model(model_path: str = "models/mental_health_model.keras") -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def predict(
    phq9_responses: list[int],
    gad7_responses: list[int],
    model: tf.keras.Model,
    scaler_path: str = "models/scaler.pkl",
) -> dict:
    """
    Predict mental health severity from questionnaire responses.

    Args:
        phq9_responses : List of 9 integers (0-3) for PHQ-9 questions.
        gad7_responses : List of 7 integers (0-3) for GAD-7 questions.
        model          : Loaded Keras model.
        scaler_path    : Path to the saved StandardScaler.

    Returns:
        dict with keys: label, confidence, phq9_score, gad7_score, probabilities
    """
    if len(phq9_responses) != 9:
        raise ValueError("PHQ-9 requires exactly 9 responses (0-3 each).")
    if len(gad7_responses) != 7:
        raise ValueError("GAD-7 requires exactly 7 responses (0-3 each).")

    raw = np.array(phq9_responses + gad7_responses, dtype=np.float32)
    scaled = scale_input(raw, scaler_path)

    probs = model.predict(scaled, verbose=0)[0]
    class_idx = int(np.argmax(probs))

    return {
        "label":       LABEL_MAP[class_idx],
        "confidence":  f"{probs[class_idx]*100:.1f}%",
        "phq9_score":  sum(phq9_responses),
        "gad7_score":  sum(gad7_responses),
        "probabilities": {LABEL_MAP[i]: f"{p*100:.1f}%" for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    # Example: moderate-severity respondent
    phq9 = [2, 1, 2, 2, 1, 2, 1, 2, 1]   # PHQ-9 total = 14
    gad7 = [2, 2, 1, 2, 1, 2, 1]          # GAD-7 total = 11

    model = load_model()
    result = predict(phq9, gad7, model)

    print("\n=== Mental Health Assessment Result ===")
    for k, v in result.items():
        print(f"  {k:<16}: {v}")
