"""
data_generator.py
Generates a synthetic dataset simulating PHQ-9 and GAD-7 questionnaire responses.

PHQ-9: 9 questions, each scored 0-3 (total 0-27) → Depression screening
GAD-7: 7 questions, each scored 0-3 (total 0-21) → Anxiety screening

Label mapping (combined score):
  0 = Minimal   (PHQ9: 0-4,  GAD7: 0-4)
  1 = Mild      (PHQ9: 5-9,  GAD7: 5-9)
  2 = Moderate  (PHQ9: 10-14, GAD7: 10-14)
  3 = Severe    (PHQ9: 15-27, GAD7: 15-21)
"""

import numpy as np
import pandas as pd


PHQ9_COLS = [f"phq9_q{i}" for i in range(1, 10)]   # 9 questions
GAD7_COLS = [f"gad7_q{i}" for i in range(1, 8)]    # 7 questions
ALL_FEATURES = PHQ9_COLS + GAD7_COLS                # 16 features total

LABEL_MAP = {0: "Minimal", 1: "Mild", 2: "Moderate", 3: "Severe"}


def _score_to_label(phq9_total: int, gad7_total: int) -> int:
    """Derive a single severity label from combined PHQ-9 and GAD-7 totals."""
    combined = phq9_total + gad7_total          # max = 27 + 21 = 48
    if combined <= 8:
        return 0   # Minimal
    elif combined <= 18:
        return 1   # Mild
    elif combined <= 30:
        return 2   # Moderate
    else:
        return 3   # Severe


def generate_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic PHQ-9 + GAD-7 dataset.

    Each row = one respondent.
    Responses are sampled from distributions biased toward each severity class
    so the dataset is realistic and balanced across all four classes.
    """
    rng = np.random.default_rng(random_state)
    records = []

    # Generate equal samples per class for a balanced dataset
    per_class = n_samples // 4

    # (low_score_mean, high_score_mean) per class to bias responses
    class_params = [
        (0.3, 0.5),   # Minimal  – mostly 0s and 1s
        (1.0, 1.2),   # Mild     – mostly 1s
        (1.8, 2.0),   # Moderate – mostly 2s
        (2.5, 2.8),   # Severe   – mostly 3s
    ]

    for label, (phq_mean, gad_mean) in enumerate(class_params):
        for _ in range(per_class):
            phq9 = np.clip(rng.normal(phq_mean, 0.6, 9).round(), 0, 3).astype(int)
            gad7 = np.clip(rng.normal(gad_mean, 0.6, 7).round(), 0, 3).astype(int)
            row = list(phq9) + list(gad7) + [label]
            records.append(row)

    df = pd.DataFrame(records, columns=ALL_FEATURES + ["label"])
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/mental_health_dataset.csv", index=False)
    print(f"Dataset saved → data/mental_health_dataset.csv")
    print(f"Shape : {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
