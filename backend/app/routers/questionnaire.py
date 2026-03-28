"""
app/routers/questionnaire.py
==============================
Questionnaire endpoints for autism risk screening.

Endpoints:
  GET  /api/v1/questionnaire/questions       — Return all 40 questions
  POST /api/v1/analyze/questionnaire         — Score responses and update session
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.session import AssessmentSession
from app.schemas.questionnaire import (
    QuestionItemSchema,
    QuestionnaireRequest,
    QuestionnaireResponse,
    QuestionsListResponse,
    SectionSchema,
)
from app.services.questionnaire_scorer import score_questionnaire
from app.utils.questionnaire_config import QUESTIONS, SCALE_LABELS, SECTIONS

logger = logging.getLogger("autisense.questionnaire")

router = APIRouter(tags=["Questionnaire"])


# ═══════════════════════════════════════════════════════════════
# GET /api/v1/questionnaire/questions
# ═══════════════════════════════════════════════════════════════


@router.get("/questionnaire/questions", response_model=QuestionsListResponse)
async def get_questions():
    """
    Return all 40 screening questions grouped by section.

    The frontend can use this to render the questionnaire dynamically
    instead of hardcoding questions.
    """
    return QuestionsListResponse(
        sections=[
            SectionSchema(
                name=s.name,
                index=s.index,
                question_count=s.end_q - s.start_q + 1,
            )
            for s in SECTIONS
        ],
        questions=[
            QuestionItemSchema(
                id=q.id,
                text=q.text,
                section=q.section,
                section_index=q.section_index,
            )
            for q in QUESTIONS
        ],
        scale=SCALE_LABELS,
        total_questions=len(QUESTIONS),
    )


# ═══════════════════════════════════════════════════════════════
# POST /api/v1/analyze/questionnaire
# ═══════════════════════════════════════════════════════════════


@router.post("/analyze/questionnaire", response_model=QuestionnaireResponse)
async def analyze_questionnaire(
    body: QuestionnaireRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Score a completed 40-question behavioral questionnaire.

    Flow:
    1. Validate request (40 responses, each 0–4, valid age, valid session)
    2. Look up session in DB
    3. Run hybrid scoring (weighted + RF model)
    4. Store results in session
    5. Return structured response

    This endpoint is synchronous — the RF model runs in milliseconds.
    """
    session_uuid = body.session_uuid

    # ── Validate session exists ─────────────────────────────────
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

    # ── Check session status allows questionnaire ───────────────
    blocked_statuses = {"error_questionnaire"}
    if session.status in blocked_statuses:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot submit questionnaire when session status is "
            f"'{session.status}'. Please create a new session.",
        )

    # Allow re-submission (idempotent) — overwrite previous results
    if session.status == "questionnaire_done":
        logger.info(
            "Re-scoring questionnaire for session %s***",
            str(session_uuid)[:8],
        )

    # ── Score the questionnaire ─────────────────────────────────
    try:
        scoring_result = score_questionnaire(
            responses=body.responses,
            child_age_months=body.child_age_months,
            child_gender=body.child_gender,
        )
    except Exception as e:
        logger.error(
            "Questionnaire scoring failed for session %s***: %s",
            str(session_uuid)[:8],
            type(e).__name__,
        )
        session.status = "error_questionnaire"
        await db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Questionnaire scoring failed. Please try again.",
        )

    # ── Update session in DB ────────────────────────────────────
    session.questionnaire_raw_scores = body.responses
    session.questionnaire_probability = scoring_result["probability"]
    session.category_scores = scoring_result["category_scores"]
    session.child_age_months = body.child_age_months
    session.child_gender = body.child_gender

    # Status transition: keep video status intact
    # If video is done, mark questionnaire_done (Sprint 6 will fuse)
    session.status = "questionnaire_done"

    await db.commit()

    logger.info(
        "Questionnaire scored: session=%s*** prob=%.4f risk=%s",
        str(session_uuid)[:8],
        scoring_result["probability"],
        scoring_result["risk_level"],
    )

    return QuestionnaireResponse(
        session_uuid=session_uuid,
        questionnaire_probability=scoring_result["probability"],
        category_scores=scoring_result["category_scores"],
        risk_level=scoring_result["risk_level"],
        status="questionnaire_done",
    )
