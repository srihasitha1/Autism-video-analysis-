
# ─────────────────────────────────────────────────────────────────
# CELL 1 — Install & Import
# ─────────────────────────────────────────────────────────────────
# !pip install scikit-learn pandas numpy --quiet   # ← Uncomment in Colab

import pandas as pd
import numpy as np
import warnings
import os
import pickle
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier   # ✅ ADD THIS

from sklearn.linear_model import LogisticRegression   # (if you use it)

from sklearn.metrics import classification_report, accuracy_score
print("✅ Libraries loaded successfully.")

# ─────────────────────────────────────────────────────────────────
# CELL 2 — Train & Save Model  (skip if model already saved)
# ─────────────────────────────────────────────────────────────────

MODEL_PATH = "autism_model.pkl"

def train_and_save_model(csv_path="/content/high_accuracy_autism_dataset.csv"):
    """Trains the model on the dataset and saves it to disk."""
    print("🔄 Training model on dataset...")

    df = pd.read_csv(csv_path)

    df = df[df["Label"].isin(["0", "1", 0, 1])].copy()
    df["Label"]  = df["Label"].astype(int)
    df["Age"]    = pd.to_numeric(df["Age"], errors="coerce")
    df = df.dropna(subset=["Age"])
    df["Gender_enc"] = df["Gender"].str.strip().map({"M": 1, "F": 0}).fillna(0).astype(int)

    question_cols = [f"Q{i}" for i in range(1, 41)]
    feature_cols  = question_cols + ["Age", "Gender_enc"]

    X = df[feature_cols].values
    y = df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print(f"📊 Cross-validation Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained!  Test Accuracy: {acc*100:.1f}%")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved to '{MODEL_PATH}'")
    return model


def load_model():
    """Load saved model from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from '{MODEL_PATH}'")
    return model


# Auto-train if model not found
if os.path.exists(MODEL_PATH):
    model = load_model()
elif os.path.exists("/content/high_accuracy_autism_dataset.csv"):
    model = train_and_save_model("/content/high_accuracy_autism_dataset.csv")
else:
    print("⚠️  No model or dataset found.")
    print("   Please upload 'questionnarie.csv' or 'autism_model.pkl' and re-run.")
    model = None


# ─────────────────────────────────────────────────────────────────
# CELL 3 — Question Bank
# ─────────────────────────────────────────────────────────────────

SECTIONS = {
    "🤝 Social Interaction": [
        "Does your child respond to their name?",
        "Does your child make eye contact?",
        "Does your child smile back when smiled at?",
        "Does your child show interest in other children?",
        "Does your child try to share enjoyment with you?",
        "Does your child point to show interest?",
        "Does your child wave goodbye?",
        "Does your child imitate actions?",
        "Does your child follow your gaze?",
        "Does your child bring objects to show you?",
    ],
    "💬 Communication": [
        "Does your child use gestures to communicate?",
        "Does your child understand simple instructions?",
        "Does your child babble or speak words?",
        "Does your child respond when spoken to?",
        "Does your child use meaningful sounds?",
        "Does your child ask for help when needed?",
        "Does your child respond to emotions?",
        "Does your child engage in back-and-forth sounds?",
        "Does your child use facial expressions?",
        "Does your child try to start conversations?",
    ],
    "🔁 Behavior Patterns": [
        "Does your child repeat actions again and again?",
        "Does your child show unusual attachment to objects?",
        "Does your child line up toys?",
        "Does your child get upset with small changes?",
        "Does your child show repetitive movements (e.g. hand flapping)?",
        "Does your child focus on parts of objects?",
        "Does your child spin or rock frequently?",
        "Does your child insist on routines?",
        "Does your child play with toys in unusual ways?",
        "Does your child show intense interest in specific things?",
    ],
    "🎯 Sensory & Emotional": [
        "Does your child react strongly to loud sounds?",
        "Does your child avoid eye contact?",
        "Does your child show limited emotional expression?",
        "Does your child overreact to touch?",
        "Does your child ignore pain or temperature?",
        "Does your child get easily frustrated?",
        "Does your child have difficulty calming down?",
        "Does your child show fear without reason?",
        "Does your child avoid social interaction?",
        "Does your child prefer to play alone?",
    ],
}

SCALE_LABEL = "  [0 = Never  |  1 = Rarely  |  2 = Sometimes  |  3 = Often  |  4 = Always]"

# ─────────────────────────────────────────────────────────────────
# CELL 4 — Interactive Questionnaire
# ─────────────────────────────────────────────────────────────────

def get_valid_int(prompt, lo=0, hi=4):
    """Keep asking until user enters a valid integer in [lo, hi]."""
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"   ⚠️  Please enter a number between {lo} and {hi}.")


def get_valid_gender():
    while True:
        g = input("   Enter Gender (M / F): ").strip().upper()
        if g in ("M", "F"):
            return 1 if g == "M" else 0
        print(f"   ⚠️  Please enter M or F.")


def run_questionnaire():
    print("\n" + "═" * 65)
    print("   🧠  AUTISM EARLY SCREENING QUESTIONNAIRE")
    print("═" * 65)
    print("  Answer each question honestly based on your child's")
    print("  typical behaviour over the past 3–6 months.")
    print("  ⚠️  DISCLAIMER: This is a screening tool only.")
    print("  It does NOT replace a professional clinical diagnosis.")
    print("═" * 65)

    # ── Basic info ──────────────────────────────────────────────
    print("\n📋  SECTION: Child Information")
    print()
    age        = get_valid_int("   Child's Age (in years, 1–6): ", lo=1, hi=6)
    gender_enc = get_valid_gender()

    # ── 40 questions across 4 sections ─────────────────────────
    responses = []
    q_number  = 1

    for section_title, questions in SECTIONS.items():
        print(f"\n{'─'*65}")
        print(f"  {section_title}")
        print(f"{'─'*65}")
        print(SCALE_LABEL)
        print()

        for question in questions:
            answer = get_valid_int(f"  Q{q_number:02d}. {question}\n       → ", lo=0, hi=4)
            responses.append(answer)
            q_number += 1

    return responses, age, gender_enc


# ─────────────────────────────────────────────────────────────────
# CELL 5 — Predict & Display Result
# ─────────────────────────────────────────────────────────────────

def display_result(probability, responses, age):
    """Print a rich, user-friendly result summary."""
    percent    = probability * 100
    prediction = "Autism Indicators Detected" if probability >= 0.5 else "No Significant Autism Indicators"

    # Section scores (average per section, scaled to %)
    section_names  = list(SECTIONS.keys())
    section_scores = []
    for i, name in enumerate(section_names):
        chunk = responses[i*10 : (i+1)*10]
        avg   = sum(chunk) / (len(chunk) * 4) * 100   # % of max possible
        section_scores.append((name, avg))

    print("\n" + "═"*65)
    print("   📊  SCREENING RESULT SUMMARY")
    print("═"*65)

    if probability < 0.3:
        flag = "🟢"
        risk = "LOW"
    elif probability < 0.6:
        flag = "🟡"
        risk = "MODERATE"
    elif probability < 0.8:
        flag = "🟠"
        risk = "ELEVATED"
    else:
        flag = "🔴"
        risk = "HIGH"

    print(f"\n  {flag}  Risk Level      : {risk}")
    print(f"  📈  Autism Probability : {percent:.1f}%")
    print(f"  🔍  Assessment         : {prediction}")
    print(f"\n{'─'*65}")
    print("  📂  SECTION BREAKDOWN  (% of maximum possible score)")
    print(f"{'─'*65}")

    for name, score in section_scores:
        filled  = int(score / 5)
        bar     = "█" * filled + "░" * (20 - filled)
        print(f"  {name}")
        print(f"     [{bar}]  {score:.1f}%")
        print()

    print("─"*65)

def run_screening(model):
    """Full pipeline: questionnaire → prediction → re4sult."""
    if model is None:
        print("❌ No model available. Please train the model first (Cell 2).")
        return

    try:
        responses, age, gender_enc = run_questionnaire()
    except KeyboardInterrupt:
        print("\n\n⚠️  Questionnaire cancelled by user.")
        return
    # ── FIX: Invert positive-behaviour questions ───────────────
    INVERT_QUESTIONS = list(range(1, 21))  # Q1–Q20 (Social + Communication)

    responses = [
        4 - r if (i + 1) in INVERT_QUESTIONS else r
        for i, r in enumerate(responses)
    ]
    # Build feature vector (Q1–Q40, Age, Gender_enc)
    # ── Section-wise scoring ─────────────────────────────
    social        = sum(responses[0:10])  / 40
    communication = sum(responses[10:20]) / 40
    behavior      = sum(responses[20:30]) / 40
    sensory       = sum(responses[30:40]) / 40

    # ── Weighted score (MAIN FIX) ───────────────────────
    weighted_score = (
        0.25 * social +
        0.25 * communication +
        0.30 * behavior +
        0.20 * sensory
    )

    # Optional: combine with model (hybrid)
    features = np.array(responses + [age, gender_enc]).reshape(1, -1)
    model_prob = model.predict_proba(features)[0][1]

    # Final probability (balanced)
    probability = (0.6 * model_prob) + (0.4 * weighted_score)
    display_result(probability, responses, age)

    # ── Ask to save ────────────────────────────────────────────
    save = input("  💾  Save this result to a file? (y/n): ").strip().lower()
    if save == "y":
        result_df = pd.DataFrame(
            [responses + [age, "M" if gender_enc == 1 else "F",
                          round(probability, 4),
                          "Low" if probability < 0.3 else
                          "Moderate" if probability < 0.6 else
                          "Elevated" if probability < 0.8 else
                          "High"]],
            columns=[f"Q{i}" for i in range(1, 41)] + ["Age", "Gender", "Probability", "Risk"]
        )
        fname = "screening_result.csv"
        result_df.to_csv(fname, index=False)
        print(f"\n  ✅  Result saved to '{fname}'")


# ─────────────────────────────────────────────────────────────────
# CELL 6 — RUN  (execute this cell to start the questionnaire)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_screening(model)