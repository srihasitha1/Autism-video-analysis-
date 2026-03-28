"""
app/services/fusion_engine.py
==============================
Intelligent Multimodal Fusion Engine — Enhanced Sprint 6.

Pure, stateless service that combines video risk scores (Sprint 4)
and questionnaire risk scores (Sprint 5) into a single final
risk assessment.

DYNAMIC WEIGHTING PHILOSOPHY:
  - Questionnaire is PRIMARY: 40 questions covering broad behavioral spectrum
  - Video is SECONDARY: Limited to 4 behavioral classes, lower coverage
  - Video must earn its influence through high confidence
  - Autism is a spectrum — decisions require comprehensive behavioral data

NO database access. NO model loading. Just maths.
"""

import logging
from typing import Any

logger = logging.getLogger("autisense.fusion")

# ── Dynamic Weighting Thresholds ──────────────────────────────────────

# Video confidence thresholds for weight selection
_CONFIDENCE_VERY_HIGH = 0.85   # ≥ 0.85: Video 50%, Q 50%
_CONFIDENCE_HIGH = 0.70        # 0.70-0.84: Video 40%, Q 60%
_CONFIDENCE_MODERATE = 0.50    # 0.50-0.69: Video 30%, Q 70%
# < 0.50: Video 20%, Q 80%

# Weight tuples: (video_weight, questionnaire_weight)
_WEIGHT_VERY_HIGH = (0.50, 0.50)   # Video confidence ≥ 0.85
_WEIGHT_HIGH = (0.40, 0.60)        # Video confidence 0.70-0.84
_WEIGHT_MODERATE = (0.30, 0.70)    # Video confidence 0.50-0.69
_WEIGHT_LOW = (0.20, 0.80)         # Video confidence < 0.50

# Fallback when video is missing
_FALLBACK_VIDEO_PROB = 0.5
_FALLBACK_VIDEO_CONFIDENCE = 0.0  # Zero confidence when no video

# Age threshold below which questionnaire is given more weight
_YOUNG_CHILD_MONTHS = 24
_AGE_SHIFT = 0.10  # +10% toward questionnaire for young children

# Risk level thresholds
_RISK_LOW_MAX = 0.35      # < 0.35 → low
_RISK_MEDIUM_MAX = 0.70   # 0.35–0.70 → medium
                           # > 0.70 → high

# Contribution concern thresholds
_CONTRIB_HIGH_MIN = 0.60
_CONTRIB_MODERATE_MIN = 0.35

# Legacy string-to-confidence mapping (backward compatibility)
_LEGACY_CONFIDENCE_MAP = {
    "high": 0.80,
    "medium": 0.60,
    "low": 0.35,
}


# ── Public API ────────────────────────────────────────────────────────


def fuse(
    questionnaire_probability: float,
    video_prob: float | None = None,
    video_confidence: str | float | None = None,
    child_age_months: int | None = None,
    video_variance: float | None = None,
) -> dict[str, Any]:
    """
    Fuse video and questionnaire risk probabilities into a final score.

    DYNAMIC WEIGHTING RULES:
    
    Video confidence measures how reliable the video prediction is.
    The questionnaire (40 questions) is always the primary source due to
    its broader behavioral coverage. Video can only gain influence by
    demonstrating high confidence.

    WEIGHT TIERS:
      - VERY HIGH (≥ 0.85): Video 50%, Questionnaire 50%
      - HIGH (0.70-0.84):   Video 40%, Questionnaire 60%
      - MODERATE (0.50-0.69): Video 30%, Questionnaire 70%
      - LOW (< 0.50):       Video 20%, Questionnaire 80%

    Args:
        questionnaire_probability: float in [0, 1] from Sprint 5 scorer.
        video_prob: float in [0, 1] from Sprint 4 video inference.
                    If None, fallback (0.5, 0 confidence) is used.
        video_confidence: Either a float in [0, 1] representing actual
                         model confidence, or legacy string "high"/"medium"/"low".
                         Floats are preferred for accurate weighting.
        child_age_months: Child age in months. If < 24, questionnaire
                          weight is increased by 10%.
        video_variance: Optional variance measure across video predictions.
                        High variance (> 0.1) reduces effective confidence.

    Returns:
        dict with keys:
          - final_risk_score: float [0, 1]
          - risk_level: "low" / "medium" / "high"
          - confidence: float [0, 1]
          - video_contribution: "high" / "moderate" / "low" concern
          - questionnaire_contribution: same
          - weights_used: {"video": float, "questionnaire": float}
          - video_fallback_used: bool
          - adjusted_video_confidence: float (after variance adjustment)
          - weighting_reasoning: str (explanation for weights chosen)
    """
    # ── Step 1: Determine if fallback is needed ──────────────────
    video_fallback_used = video_prob is None
    if video_fallback_used:
        video_prob = _FALLBACK_VIDEO_PROB
        video_confidence = _FALLBACK_VIDEO_CONFIDENCE
        logger.debug("No video data — using fallback: prob=0.5, confidence=0.0")

    # ── Step 2: Normalize confidence to numeric ──────────────────
    raw_confidence = _normalize_confidence(video_confidence)
    adjusted_confidence = raw_confidence

    # ── Step 3: Adjust confidence for prediction variance ───────
    variance_penalty = 0.0
    if video_variance is not None and video_variance > 0.1:
        # High variance indicates inconsistent predictions
        # Reduce effective confidence proportionally
        variance_penalty = min(0.2, video_variance * 0.5)
        adjusted_confidence = max(0.0, raw_confidence - variance_penalty)
        logger.debug(
            "High video variance (%.4f) — confidence reduced: %.4f → %.4f",
            video_variance, raw_confidence, adjusted_confidence,
        )

    # ── Step 4: Select weights based on adjusted confidence ──────
    w_video, w_q, tier_name = _select_weights(adjusted_confidence)
    weighting_reasoning = _build_reasoning(
        tier_name, adjusted_confidence, raw_confidence, variance_penalty
    )

    # ── Step 5: Age adjustment ───────────────────────────────────
    # Very young children's behaviour is harder to capture on video.
    # Shift weight +10% toward questionnaire (parental observation).
    if child_age_months is not None and child_age_months < _YOUNG_CHILD_MONTHS:
        w_q = min(1.0, w_q + _AGE_SHIFT)
        w_video = max(0.0, 1.0 - w_q)
        weighting_reasoning += f" Age < 24mo (+10% to questionnaire)."
        logger.debug(
            "Age < 24 months (%d) — weights adjusted: video=%.2f q=%.2f",
            child_age_months,
            w_video,
            w_q,
        )

    # ── Step 6: Weighted combination ────────────────────────────
    raw_score = (w_video * video_prob) + (w_q * questionnaire_probability)
    final_risk_score = max(0.0, min(1.0, raw_score))  # clamp to [0, 1]

    # ── Step 7: Risk level classification ───────────────────────
    risk_level = _classify_risk(final_risk_score)

    # ── Step 8: Overall confidence score ─────────────────────────
    # Distance from 0.5, scaled to [0, 1].
    # A score near 0.5 means uncertain; near 0 or 1 means very clear.
    confidence = abs(final_risk_score - 0.5) * 2.0

    # ── Step 9: Contribution labels (per modality) ───────────────
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
        "adjusted_video_confidence": round(adjusted_confidence, 4),
        "weighting_reasoning": weighting_reasoning,
    }

    logger.info(
        "Fusion: score=%.4f risk=%s confidence=%.4f "
        "weights=[v=%.2f q=%.2f] video_conf=%.4f tier=%s",
        final_risk_score,
        risk_level,
        confidence,
        w_video,
        w_q,
        adjusted_confidence,
        tier_name,
    )

    return result


# ── Private helpers ──────────────────────────────────────────────────────


def _normalize_confidence(confidence: str | float | None) -> float:
    """
    Convert confidence to a numeric value in [0, 1].
    
    Handles legacy string values ("high", "medium", "low") for backward
    compatibility with existing video inference output.
    """
    if confidence is None:
        return _LEGACY_CONFIDENCE_MAP["low"]
    
    if isinstance(confidence, (int, float)):
        return max(0.0, min(1.0, float(confidence)))
    
    # Legacy string mapping
    key = str(confidence).lower()
    return _LEGACY_CONFIDENCE_MAP.get(key, _LEGACY_CONFIDENCE_MAP["low"])


def _select_weights(adjusted_confidence: float) -> tuple[float, float, str]:
    """
    Select weights based on adjusted video confidence.
    
    Returns:
        (video_weight, questionnaire_weight, tier_name)
    """
    if adjusted_confidence >= _CONFIDENCE_VERY_HIGH:
        return (*_WEIGHT_VERY_HIGH, "VERY_HIGH")
    elif adjusted_confidence >= _CONFIDENCE_HIGH:
        return (*_WEIGHT_HIGH, "HIGH")
    elif adjusted_confidence >= _CONFIDENCE_MODERATE:
        return (*_WEIGHT_MODERATE, "MODERATE")
    else:
        return (*_WEIGHT_LOW, "LOW")


def _build_reasoning(
    tier_name: str,
    adjusted_confidence: float,
    raw_confidence: float,
    variance_penalty: float,
) -> str:
    """Build human-readable explanation for weighting decision."""
    tier_descriptions = {
        "VERY_HIGH": "Video confidence very high (≥0.85). Equal weighting applied.",
        "HIGH": "Video confidence high (0.70-0.84). Questionnaire weighted higher.",
        "MODERATE": "Video confidence moderate (0.50-0.69). Questionnaire strongly favored.",
        "LOW": "Video confidence low (<0.50). Questionnaire dominant (80%).",
    }
    
    reasoning = tier_descriptions.get(tier_name, "Default weighting applied.")
    
    if variance_penalty > 0:
        reasoning += f" Confidence reduced by {variance_penalty:.2f} due to prediction variance."
    
    return reasoning


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
