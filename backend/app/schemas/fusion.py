"""
app/schemas/fusion.py
======================
Pydantic schemas for the Fusion Engine & Risk Report API (Sprint 6).

Covers:
  POST /api/v1/analyze/fuse
  GET  /api/v1/analyze/report/{uuid}
  GET  /api/v1/analyze/report/{uuid}/summary
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel

_DISCLAIMER = (
    "This is an early behavioral risk assessment tool, not a diagnostic instrument. "
    "Please consult a qualified healthcare professional for clinical evaluation."
)


# ── POST /analyze/fuse ──────────────────────────────────────────


class FuseRequest(BaseModel):
    """Request body for the fusion endpoint."""

    session_uuid: UUID


class FuseResponse(BaseModel):
    """Response after successful fusion."""

    session_uuid: UUID
    final_risk_score: float
    risk_level: str                          # "low" / "medium" / "high"
    confidence: float                        # 0–1, distance from 0.5
    video_contribution: str                  # "high/moderate/low concern"
    questionnaire_contribution: str
    weights_used: dict[str, float]           # {"video": 0.5, "questionnaire": 0.5}
    video_fallback_used: bool
    adjusted_video_confidence: Optional[float] = None  # Confidence after variance adjustment
    weighting_reasoning: Optional[str] = None          # Explanation for weights chosen
    status: str                              # always "complete"
    disclaimer: str = _DISCLAIMER


# ── GET /analyze/report/{uuid} ──────────────────────────────────


class RiskReportResponse(BaseModel):
    """Full risk report — all assessment data for a completed session."""

    session_uuid: UUID
    status: str

    # ── Fused results ───────────────────────────────────────────
    final_risk_score: float
    risk_level: str
    confidence: float
    video_contribution: str
    questionnaire_contribution: str
    weights_used: dict[str, float]
    video_fallback_used: bool

    # ── Source data (read-only, not recomputed) ─────────────────
    video_score: Optional[float] = None
    video_confidence: Optional[str] = None
    video_class_probabilities: Optional[dict[str, Any]] = None
    questionnaire_probability: Optional[float] = None
    category_scores: Optional[dict[str, float]] = None

    # ── Child info ──────────────────────────────────────────────
    child_age_months: Optional[int] = None
    child_gender: Optional[str] = None

    # ── Timestamps ──────────────────────────────────────────────
    created_at: Optional[datetime] = None

    disclaimer: str = _DISCLAIMER


# ── GET /analyze/report/{uuid}/summary ──────────────────────────


class RiskReportSummary(BaseModel):
    """
    Lightweight risk summary — for frontend dashboard cards.

    Only includes the essential fields needed to render a summary card.
    Does not include raw model outputs or detailed breakdowns.
    """

    session_uuid: UUID
    final_risk_score: float
    risk_level: str
    confidence: float
    video_contribution: str
    questionnaire_contribution: str
    video_fallback_used: bool
    status: str
    disclaimer: str = _DISCLAIMER
