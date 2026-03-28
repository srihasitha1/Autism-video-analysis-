"""
app/services/fusion_engine.py
==============================
Multimodal Fusion Engine — Sprint 6.

Pure, stateless service that combines video risk scores (Sprint 4)
and questionnaire risk scores (Sprint 5) into a single final
risk assessment.

NO database access. NO model loading. Just maths.

Design:
  - Assign base weights from video confidence level
  - Adjust weights for very young children (< 24 months)
  - Compute weighted probability, clamped to [0, 1]
  - Map to risk level, confidence, and contribution labels
"""

import logging
from typing import Any

logger = logging.getLogger("autisense.fusion")

# ── Constants ────────────────────────────────────────────────────

# Base weights (video_weight, questionnaire_weight)
_BASE_WEIGHTS: dict[str, tuple[float, float]] = {
    "high":   (0.60, 0.40),
    "medium": (0.50, 0.50),
    "low":    (0.30, 0.70),
}

# When video is missing, treat as low-confidence fallback
_FALLBACK_VIDEO_PROB = 0.5
_FALLBACK_VIDEO_CONFIDENCE = "low"

# Age threshold below which questionnaire is given more weight
_YOUNG_CHILD_MONTHS = 24
_AGE_SHIFT = 0.10  # +10% toward questionnaire for young children

# Risk level thresholds
_RISK_LOW_MAX = 0.35      # < 0.35  → low
_RISK_MEDIUM_MAX = 0.70   # 0.35–0.70 → medium
                           # > 0.70  → high

# Contribution concern thresholds
_CONTRIB_HIGH_MIN = 0.60
_CONTRIB_MODERATE_MIN = 0.35


# ── Public API ───────────────────────────────────────────────────


def fuse(
    questionnaire_probability: float,
    video_prob: float | None = None,
    video_confidence: str | None = None,
    child_age_months: int | None = None,
) -> dict[str, Any]:
    """
    Fuse video and questionnaire risk probabilities into a final score.

    Args:
        questionnaire_probability: float in [0, 1] from Sprint 5 scorer.
        video_prob: float in [0, 1] from Sprint 4 video inference.
                    If None, the fallback (0.5, low confidence) is used.
        video_confidence: "high" / "medium" / "low". None treated as "low".
        child_age_months: Child age in months. If < 24, questionnaire
                          weight is increased by 10%.

    Returns:
        dict with keys:
          - final_risk_score: float [0, 1]
          - risk_level: "low" / "medium" / "high"
          - confidence: float [0, 1]
          - video_contribution: "high" / "moderate" / "low" concern
          - questionnaire_contribution: same
          - weights_used: {"video": float, "questionnaire": float}
          - video_fallback_used: bool
    """
    # ── Step 1: Determine if fallback is needed ──────────────────
    video_fallback_used = video_prob is None
    if video_fallback_used:
        video_prob = _FALLBACK_VIDEO_PROB
        video_confidence = _FALLBACK_VIDEO_CONFIDENCE
        logger.debug("No video data — using fallback: prob=0.5, confidence=low")

    # ── Step 2: Lookup base weights ──────────────────────────────
    confidence_key = (video_confidence or "low").lower()
    if confidence_key not in _BASE_WEIGHTS:
        confidence_key = "low"

    w_video, w_q = _BASE_WEIGHTS[confidence_key]

    # ── Step 3: Age adjustment ───────────────────────────────────
    # Very young children's behaviour is harder to capture on video.
    # Shift weight +10% toward questionnaire (parental observation).
    if child_age_months is not None and child_age_months < _YOUNG_CHILD_MONTHS:
        w_q = min(1.0, w_q + _AGE_SHIFT)
        w_video = max(0.0, 1.0 - w_q)
        logger.debug(
            "Age < 24 months (%d) — weights adjusted: video=%.2f q=%.2f",
            child_age_months,
            w_video,
            w_q,
        )

    # ── Step 4: Weighted combination ────────────────────────────
    raw_score = (w_video * video_prob) + (w_q * questionnaire_probability)
    final_risk_score = max(0.0, min(1.0, raw_score))  # clamp to [0, 1]

    # ── Step 5: Risk level classification ───────────────────────
    risk_level = _classify_risk(final_risk_score)

    # ── Step 6: Confidence score ─────────────────────────────────
    # Distance from 0.5, scaled to [0, 1].
    # A score near 0.5 means uncertain; near 0 or 1 means very clear.
    confidence = abs(final_risk_score - 0.5) * 2.0

    # ── Step 7: Contribution labels (per modality) ───────────────
    video_contribution = _classify_contribution(video_prob)
    questionnaire_contribution = _classify_contribution(questionnaire_probability)

    result = {
        "final_risk_score": round(final_risk_score, 4),
        "risk_level": risk_level,
        "confidence": round(confidence, 4),
        "video_contribution": video_contribution,
        "questionnaire_contribution": questionnaire_contribution,
        "weights_used": {
            "video": round(w_video, 2),
            "questionnaire": round(w_q, 2),
        },
        "video_fallback_used": video_fallback_used,
    }

    logger.info(
        "Fusion complete: score=%.4f risk=%s confidence=%.4f "
        "weights=[v=%.2f q=%.2f] fallback=%s",
        final_risk_score,
        risk_level,
        confidence,
        w_video,
        w_q,
        video_fallback_used,
    )

    return result


# ── Private helpers ──────────────────────────────────────────────


def _classify_risk(score: float) -> str:
    """Map a probability to a risk level string."""
    if score < _RISK_LOW_MAX:
        return "low"
    elif score <= _RISK_MEDIUM_MAX:
        return "medium"
    else:
        return "high"


def _classify_contribution(prob: float) -> str:
    """
    Label the concern level of a single modality's probability.

    Returns "high concern", "moderate concern", or "low concern".
    """
    if prob >= _CONTRIB_HIGH_MIN:
        return "high concern"
    elif prob >= _CONTRIB_MODERATE_MIN:
        return "moderate concern"
    else:
        return "low concern"
