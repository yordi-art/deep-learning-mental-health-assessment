# Mental Health Assessment AI

A deep learning model that classifies mental health severity from **PHQ-9** (depression) and **GAD-7** (anxiety) questionnaire responses using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

---

## Problem Statement

Mental health conditions like depression and anxiety are often under-diagnosed. Standard clinical tools (PHQ-9, GAD-7) produce numerical scores that can be automatically classified using machine learning — enabling faster, scalable screening support.

> ⚠️ **Disclaimer:** This tool is for educational/research purposes only. It is **not** a substitute for professional medical diagnosis.

---

## Model Output

| Class | PHQ-9 + GAD-7 Combined Score |
|-------|------------------------------|
| Minimal  | 0 – 8   |
| Mild     | 9 – 18  |
| Moderate | 19 – 30 |
| Severe   | 31 – 48 |

---

## Project Structure

```
mental-health-assessment-ai/
├── data/
│   └── mental_health_dataset.csv     # Auto-generated synthetic dataset
├── models/
│   ├── mental_health_model.keras     # Saved trained model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   ├── training_curves.png           # Loss & accuracy plots
│   └── confusion_matrix.png          # Evaluation heatmap
├── src/
│   ├── data_generator.py             # Synthetic PHQ-9 / GAD-7 data generation
│   ├── preprocessor.py               # Scaling, splitting, scaler persistence
│   ├── model.py                      # ANN architecture (Keras Functional API)
│   ├── trainer.py                    # Training loop with callbacks
│   ├── evaluator.py                  # Metrics, plots, confusion matrix
│   └── predictor.py                  # Inference on new user input
├── notebooks/
│   └── exploration.ipynb             # (optional) EDA and experimentation
├── train.py                          # Main entry point
├── requirements.txt
└── README.md
```

---

## Model Architecture

```
Input (16 features)
    │
    ▼
Dense(128, ReLU) + BatchNorm + Dropout(0.3)
    │
    ▼
Dense(64, ReLU)  + BatchNorm + Dropout(0.3)
    │
    ▼
Dense(32, ReLU)  + BatchNorm + Dropout(0.2)
    │
    ▼
Dense(4, Softmax)  ← [Minimal, Mild, Moderate, Severe]
```

- **16 input features** — 9 PHQ-9 + 7 GAD-7 question scores (each 0–3)
- **BatchNormalization** — stabilizes and speeds up training
- **Dropout** — regularization to prevent overfitting
- **L2 weight regularization** — penalizes large weights
- **Optimizer:** Adam | **Loss:** Sparse Categorical Crossentropy

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This will:
- Generate a synthetic dataset (2000 samples) in `data/`
- Train the ANN with early stopping
- Save the model and scaler to `models/`
- Print test accuracy and classification report
- Save training curves and confusion matrix plots

### 3. Run inference on new input

```bash
python -m src.predictor
```

Or use it programmatically:

```python
from src.predictor import load_model, predict

model = load_model()

phq9 = [2, 1, 2, 2, 1, 2, 1, 2, 1]  # 9 responses (0–3)
gad7 = [2, 2, 1, 2, 1, 2, 1]         # 7 responses (0–3)

result = predict(phq9, gad7, model)
print(result)
# {
#   "label": "Moderate",
#   "confidence": "87.3%",
#   "phq9_score": 14,
#   "gad7_score": 11,
#   "probabilities": {"Minimal": "1.2%", "Mild": "8.4%", "Moderate": "87.3%", "Severe": "3.1%"}
# }
```

---

## Questionnaire Reference

### PHQ-9 Questions (Depression Screening)
Each question: *"Over the last 2 weeks, how often have you been bothered by..."*

| # | Question |
|---|----------|
| 1 | Little interest or pleasure in doing things |
| 2 | Feeling down, depressed, or hopeless |
| 3 | Trouble falling/staying asleep, or sleeping too much |
| 4 | Feeling tired or having little energy |
| 5 | Poor appetite or overeating |
| 6 | Feeling bad about yourself |
| 7 | Trouble concentrating |
| 8 | Moving/speaking slowly or being fidgety/restless |
| 9 | Thoughts of self-harm |

### GAD-7 Questions (Anxiety Screening)
Each question: *"Over the last 2 weeks, how often have you been bothered by..."*

| # | Question |
|---|----------|
| 1 | Feeling nervous, anxious, or on edge |
| 2 | Not being able to stop or control worrying |
| 3 | Worrying too much about different things |
| 4 | Trouble relaxing |
| 5 | Being so restless it's hard to sit still |
| 6 | Becoming easily annoyed or irritable |
| 7 | Feeling afraid something awful might happen |

**Scoring:** 0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model building & training |
| scikit-learn | Preprocessing, metrics |
| pandas / numpy | Data handling |
| matplotlib / seaborn | Visualization |
| joblib | Scaler persistence |

---

## License

MIT License — free to use for research and educational purposes.
