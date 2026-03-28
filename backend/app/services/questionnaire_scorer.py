"""
app/services/questionnaire_scorer.py
=====================================
Hybrid questionnaire scoring service.

Replicates the scoring logic from model/questionnarie_model_test.py:
  1. Invert Q1–Q20 (positive-behaviour → risk scores)
  2. Compute section-wise scores (social, communication, behavior, sensory)
  3. Compute weighted aggregate score
  4. Run Random Forest model prediction
  5. Combine: (0.6 × model_prob) + (0.4 × weighted_score)

The RF model is loaded once (singleton) following the same pattern
as video_inference.py.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.utils.questionnaire_config import INVERT_INDICES, SECTIONS

logger = logging.getLogger("autisense.questionnaire")

# ── Module-level singleton cache ────────────────────────────────
_rf_model = None
_model_loaded = False
_model_available = False


def _load_rf_model():
    """Load the Random Forest model from disk. Cached as singleton."""
    global _rf_model, _model_loaded, _model_available

    if _model_loaded:
        return _rf_model

    model_path = str(Path(settings.MODEL_RF_PATH).resolve())

    if not os.path.exists(model_path):
        logger.warning(
            "Questionnaire RF model not found at %s — "
            "falling back to weighted-only scoring.",
            model_path,
        )
        _model_loaded = True
        _model_available = False
        return None

    try:
        with open(model_path, "rb") as f:
            _rf_model = pickle.load(f)
        _model_loaded = True
        _model_available = True
        logger.info("Questionnaire RF model loaded from: %s", Path(model_path).name)
        return _rf_model
    except Exception as e:
        logger.error("Failed to load questionnaire model: %s", e)
        _model_loaded = True
        _model_available = False
        return None


def reset_model_cache():
    """Reset the singleton cache. Used in tests."""
    global _rf_model, _model_loaded, _model_available
    _rf_model = None
    _model_loaded = False
    _model_available = False


def _invert_responses(responses: list[int]) -> list[int]:
    """
    Invert positive-behaviour questions (Q1–Q20).

    For these questions, a high score (4 = Always) means the child
    exhibits the positive behaviour → low autism risk. We invert
    so that high values indicate higher risk across all questions.
    """
    return [
        4 - r if i in INVERT_INDICES else r
        for i, r in enumerate(responses)
    ]


def _compute_category_scores(inverted: list[int]) -> dict[str, float]:
    """
    Compute per-section risk scores (0.0–1.0 scale).

    Each section has 10 questions scored 0–4.
    Section score = sum(section_responses) / 40
    (This normalises to 0–1 range: max is 10×4=40.)
    """
    scores: dict[str, float] = {}
    for section in SECTIONS:
        start = (section.index) * 10
        end = start + 10
        section_sum = sum(inverted[start:end])
        # Normalise: max possible = 10 questions × 4 = 40
        scores[section.name.lower().replace(" & ", "_").replace(" ", "_")] = round(
            section_sum / 40.0, 4
        )
    return scores


def _classify_risk(probability: float) -> str:
    """Classify autism probability into risk levels."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    elif probability < 0.8:
        return "Elevated"
    else:
        return "High"


def _encode_gender(gender: str | None) -> int:
    """Encode gender string to numeric value matching training data."""
    if gender is None or gender.lower() == "unspecified":
        return 0  # Default
    return 1 if gender.lower() == "male" else 0


def score_questionnaire(
    responses: list[int],
    child_age_months: int,
    child_gender: str | None = None,
) -> dict[str, Any]:
    """
    Score a completed 40-question questionnaire.

    Args:
        responses: List of 40 integers, each in [0, 4].
        child_age_months: Child's age in months (12–96).
        child_gender: "male", "female", or "unspecified" / None.

    Returns:
        dict with:
          - probability: float (0–1, final hybrid score)
          - category_scores: dict (per-section scores)
          - risk_level: str ("Low"/"Moderate"/"Elevated"/"High")
          - model_used: bool (whether RF model was available)
    """
    if len(responses) != 40:
        raise ValueError(f"Expected 40 responses, got {len(responses)}")

    # ── Step 1: Invert positive-behaviour questions ─────────────
    inverted = _invert_responses(responses)

    # ── Step 2: Compute per-section scores ──────────────────────
    category_scores = _compute_category_scores(inverted)

    # ── Step 3: Weighted aggregate score ────────────────────────
    social = category_scores.get("social_interaction", 0.0)
    communication = category_scores.get("communication", 0.0)
    behavior = category_scores.get("behavior_patterns", 0.0)
    sensory = category_scores.get("sensory_emotional", 0.0)

    weighted_score = (
        0.25 * social
        + 0.25 * communication
        + 0.30 * behavior
        + 0.20 * sensory
    )

    # ── Step 4: RF model prediction ─────────────────────────────
    model = _load_rf_model()
    gender_enc = _encode_gender(child_gender)

    # Convert age from months to years (model was trained on age in years)
    age_years = max(1, child_age_months // 12)

    if model is not None:
        # Build 42-feature vector: [Q1..Q40 (inverted), Age, Gender_enc]
        features = np.array(
            inverted + [age_years, gender_enc], dtype=float
        ).reshape(1, -1)

        model_prob = float(model.predict_proba(features)[0][1])

        # ── Step 5: Hybrid combination ──────────────────────────
        probability = (0.6 * model_prob) + (0.4 * weighted_score)
        model_used = True
    else:
        # Fallback: weighted-only scoring
        probability = weighted_score
        model_used = False

    # Clamp to [0, 1]
    probability = max(0.0, min(1.0, round(probability, 4)))

    # ── Step 6: Risk classification ─────────────────────────────
    risk_level = _classify_risk(probability)

    logger.info(
        "Questionnaire scored: prob=%.4f risk=%s model_used=%s",
        probability,
        risk_level,
        model_used,
    )

    return {
        "probability": probability,
        "category_scores": category_scores,
        "risk_level": risk_level,
        "model_used": model_used,
    }
