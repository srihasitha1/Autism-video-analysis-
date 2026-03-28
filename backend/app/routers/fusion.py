"""
app/routers/fusion.py
======================
Fusion Engine & Risk Report endpoints — Sprint 6.

Endpoints:
  POST /api/v1/analyze/fuse              — Run fusion, set status="complete"
  GET  /api/v1/analyze/report/{uuid}    — Full risk report
  GET  /api/v1/analyze/report/{uuid}/summary — Dashboard card summary
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.session import AssessmentSession
from app.schemas.fusion import FuseRequest, FuseResponse, RiskReportResponse, RiskReportSummary
from app.services.fusion_engine import fuse

logger = logging.getLogger("autisense.fusion")

router = APIRouter(tags=["Fusion & Reports"])


# ═══════════════════════════════════════════════════════════════
# POST /api/v1/analyze/fuse
# ═══════════════════════════════════════════════════════════════


@router.post("/analyze/fuse", response_model=FuseResponse)
async def fuse_analysis(
    body: FuseRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Fuse video + questionnaire results into a final risk assessment.

    Flow:
    1. Look up session
    2. Validate questionnaire data is present (required)
    3. Use video data if available, otherwise apply fallback
    4. Run fusion engine (pure computation, no DB inside)
    5. Write fusion results + set status="complete"
    6. Return FuseResponse

    Idempotent: re-fusing a complete session overwrites with the same result.
    """
    session_uuid = body.session_uuid

    # ── Look up session ──────────────────────────────────────────
    result = await db.execute(
        select(AssessmentSession).where(
            AssessmentSession.session_uuid == session_uuid
        )
    )
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # ── Questionnaire is required ────────────────────────────────
    if session.questionnaire_probability is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Questionnaire must be completed before fusion. "
                "Please submit questionnaire responses first."
            ),
        )

    # ── Log re-fusion ────────────────────────────────────────────
    if session.status == "complete":
        logger.info(
            "Re-fusing already-complete session %s***",
            str(session_uuid)[:8],
        )

    # ── Run fusion (video is optional) ───────────────────────────
    fusion_result = fuse(
        questionnaire_probability=session.questionnaire_probability,
        video_prob=session.video_score,            # None if no video
        video_confidence=session.video_confidence_score,  # Numeric confidence (preferred)
        child_age_months=session.child_age_months, # None if not provided
        video_variance=session.video_variance,     # For variance-based adjustment
    )

    # ── Write results to session ─────────────────────────────────
    session.final_risk_score = fusion_result["final_risk_score"]
    session.risk_level = fusion_result["risk_level"]
    session.confidence = fusion_result["confidence"]
    session.video_contribution = fusion_result["video_contribution"]
    session.questionnaire_contribution = fusion_result["questionnaire_contribution"]
    session.status = "complete"

    await db.commit()

    logger.info(
        "Fusion complete: session=%s*** score=%.4f risk=%s",
        str(session_uuid)[:8],
        fusion_result["final_risk_score"],
        fusion_result["risk_level"],
    )

    return FuseResponse(
        session_uuid=session_uuid,
        final_risk_score=fusion_result["final_risk_score"],
        risk_level=fusion_result["risk_level"],
        confidence=fusion_result["confidence"],
        video_contribution=fusion_result["video_contribution"],
        questionnaire_contribution=fusion_result["questionnaire_contribution"],
        weights_used=fusion_result["weights_used"],
        video_fallback_used=fusion_result["video_fallback_used"],
        adjusted_video_confidence=fusion_result.get("adjusted_video_confidence"),
        weighting_reasoning=fusion_result.get("weighting_reasoning"),
        status="complete",
    )


# ═══════════════════════════════════════════════════════════════
# GET /api/v1/analyze/report/{uuid}
# ═══════════════════════════════════════════════════════════════


@router.get("/analyze/report/{session_uuid}", response_model=RiskReportResponse)
async def get_full_report(
    session_uuid: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Return the full risk assessment report.

    Only available when session status is "complete".
    Returns all source data alongside fusion results for complete
    frontend rendering (detailed breakdown view).
    """
    session = await _get_complete_session(session_uuid, db)

    return RiskReportResponse(
        session_uuid=session_uuid,
        status=session.status,
        # Fused results
        final_risk_score=session.final_risk_score,
        risk_level=session.risk_level,
        confidence=session.confidence,
        video_contribution=session.video_contribution,
        questionnaire_contribution=session.questionnaire_contribution,
        weights_used=_reconstruct_weights(session),
        video_fallback_used=session.video_score is None,
        # Source data
        video_score=session.video_score,
        video_confidence=session.video_confidence,
        video_class_probabilities=session.video_class_probabilities,
        questionnaire_probability=session.questionnaire_probability,
        category_scores=session.category_scores,
        # Child info
        child_age_months=session.child_age_months,
        child_gender=session.child_gender,
        # Timestamps
        created_at=session.created_at,
    )


# ═══════════════════════════════════════════════════════════════
# GET /api/v1/analyze/report/{uuid}/summary
# ═══════════════════════════════════════════════════════════════


@router.get("/analyze/report/{session_uuid}/summary", response_model=RiskReportSummary)
async def get_report_summary(
    session_uuid: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Return a lightweight risk summary for dashboard cards.

    Only available when session status is "complete".
    Returns only the essential fields needed to render a summary card
    without exposing raw model outputs.
    """
    session = await _get_complete_session(session_uuid, db)

    return RiskReportSummary(
        session_uuid=session_uuid,
        final_risk_score=session.final_risk_score,
        risk_level=session.risk_level,
        confidence=session.confidence,
        video_contribution=session.video_contribution,
        questionnaire_contribution=session.questionnaire_contribution,
        video_fallback_used=session.video_score is None,
        status=session.status,
    )


# ── Private helpers ──────────────────────────────────────────────


async def _get_complete_session(
    session_uuid: UUID,
    db: AsyncSession,
) -> AssessmentSession:
    """Fetch session and assert it is complete. Raises HTTPException otherwise."""
    result = await db.execute(
        select(AssessmentSession).where(
            AssessmentSession.session_uuid == session_uuid
        )
    )
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.status != "complete":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Report not ready: assessment not complete. "
                f"Current status is '{session.status}'."
            ),
        )

    return session


def _reconstruct_weights(session: AssessmentSession) -> dict[str, float]:
    """
    Reconstruct the weights_used dict for the report.

    Uses the dynamic weighting logic from fusion_engine based on
    numeric video confidence score. Falls back to legacy string-based
    confidence if numeric score is not available.
    """
    from app.services.fusion_engine import (
        _select_weights,
        _normalize_confidence,
        _AGE_SHIFT,
        _YOUNG_CHILD_MONTHS,
    )

    # Prefer numeric confidence score, fall back to legacy string
    if session.video_confidence_score is not None:
        confidence = session.video_confidence_score
    else:
        confidence = _normalize_confidence(session.video_confidence)

    w_video, w_q, _ = _select_weights(confidence)

    if session.child_age_months is not None and session.child_age_months < _YOUNG_CHILD_MONTHS:
        w_q = min(1.0, w_q + _AGE_SHIFT)
        w_video = max(0.0, 1.0 - w_q)

    return {"video": round(w_video, 2), "questionnaire": round(w_q, 2)}
