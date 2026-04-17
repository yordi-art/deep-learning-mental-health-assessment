"""
app.py
Streamlit web app for Mental Health Assessment AI.
Uses the trained ANN model to classify PHQ-9 + GAD-7 responses.
"""

import streamlit as st
import numpy as np
from src.predictor import load_model, predict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Assessment AI",
    page_icon="🧠",
    layout="centered",
)

# ── Load model once (cached so it doesn't reload on every interaction) ────────
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ── Recommendation map ────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Minimal":  ("🟢 Minimal", "Self-care and wellness practices are sufficient. Consider mindfulness or journaling."),
    "Mild":     ("🟡 Mild",    "A **Counselor** or **Life Coach** can help with coping strategies."),
    "Moderate": ("🟠 Moderate","A licensed **Therapist** (CBT/talk therapy) is recommended."),
    "Severe":   ("🔴 Severe",  "Please consult a **Clinical Psychologist** or **Psychiatrist** promptly."),
}

# ── PHQ-9 and GAD-7 questions ─────────────────────────────────────────────────
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling/staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself",
    "Trouble concentrating on things",
    "Moving/speaking slowly — or being fidgety/restless",
    "Thoughts of self-harm or being better off dead",
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless it's hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid something awful might happen",
]

SCORE_LABELS = {0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Mental Health Assessment AI")
st.markdown(
    "Answer the questions below based on how you've felt **over the last 2 weeks**. "
    "Then click **Predict** to see your result."
)


# ── PHQ-9 Section ─────────────────────────────────────────────────────────────
st.subheader(" PHQ-9 — Depression Screening")
st.caption("How often have you been bothered by the following?")

phq9_responses = []
for i, question in enumerate(PHQ9_QUESTIONS):
    val = st.slider(
        label=f"**{i+1}.** {question}",
        min_value=0, max_value=3, value=0,
        format="%d",
        key=f"phq9_{i}",
        help=" | ".join([f"{k}={v}" for k, v in SCORE_LABELS.items()])
    )
    phq9_responses.append(val)

st.divider()

# ── GAD-7 Section ─────────────────────────────────────────────────────────────
st.subheader("GAD-7 — Anxiety Screening")
st.caption("How often have you been bothered by the following?")

gad7_responses = []
for i, question in enumerate(GAD7_QUESTIONS):
    val = st.slider(
        label=f"**{i+1}.** {question}",
        min_value=0, max_value=3, value=0,
        format="%d",
        key=f"gad7_{i}",
        help=" | ".join([f"{k}={v}" for k, v in SCORE_LABELS.items()])
    )
    gad7_responses.append(val)

st.divider()

# ── Score summary (live) ──────────────────────────────────────────────────────
phq9_total = sum(phq9_responses)
gad7_total  = sum(gad7_responses)

col1, col2, col3 = st.columns(3)
col1.metric("PHQ-9 Score",    f"{phq9_total} / 27")
col2.metric("GAD-7 Score",    f"{gad7_total} / 21")
col3.metric("Combined Score", f"{phq9_total + gad7_total} / 48")

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    with st.spinner("Analyzing responses..."):
        result = predict(phq9_responses, gad7_responses, model)

    label      = result["label"]
    confidence = result["confidence"]
    probs      = result["probabilities"]
    badge, advice = RECOMMENDATIONS[label]

    # Result card
    st.success(f"### Assessment Result: {badge}")
    st.markdown(f"**Confidence:** {confidence}")
    st.markdown(f"**Recommendation:** {advice}")

    st.divider()

    # Probability breakdown
    st.subheader("📊 Probability Breakdown")
    for cls, prob in probs.items():
        pct = float(prob.replace("%", ""))
        st.progress(int(pct), text=f"{cls}: {prob}")
