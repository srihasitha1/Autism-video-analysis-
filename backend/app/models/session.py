"""
app/models/session.py
=====================
ORM model for assessment sessions.

HARD RULE: No PII columns (name, email, DOB, IP, phone, address) — ever.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    func,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSON, UUID

from app.db.base import Base


class AssessmentSession(Base):
    """
    Stores a single anonymous assessment session.
    Each session tracks video analysis + questionnaire results + fused risk score.
    No personally identifiable information is stored.
    """

    __tablename__ = "assessment_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_uuid = Column(
        UUID(as_uuid=True),
        unique=True,
        nullable=False,
        default=uuid.uuid4,
        index=True,
    )

    # Link to registered user (nullable for guest sessions)
    user_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_uuid", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Child info (minimal, non-identifying) ───────────────────
    child_age_months = Column(Integer, nullable=True)
    child_gender = Column(String(20), nullable=True)  # "male"/"female"/"unspecified"

    # ── Video analysis results ──────────────────────────────────
    video_class_probabilities = Column(JSON, nullable=True)
    # → {"arm_flapping": 0.72, "spinning": 0.15, "head_banging": 0.08, "normal": 0.65}
    video_confidence = Column(String(20), nullable=True)  # "high"/"medium"/"low" (legacy)
    video_confidence_score = Column(Float, nullable=True)  # Numeric confidence 0–1 for dynamic weighting
    video_variance = Column(Float, nullable=True)  # Prediction variance across clips
    video_score = Column(Float, nullable=True)  # Raw autism score from model (0–1)
    video_error = Column(String(500), nullable=True)  # Error message on failure
    celery_task_id = Column(String(255), nullable=True)  # Celery AsyncResult ID

    # ── Questionnaire results ───────────────────────────────────
    questionnaire_raw_scores = Column(JSON, nullable=True)  # [0,1,2,3,4,...] (40 ints)
    questionnaire_probability = Column(Float, nullable=True)
    category_scores = Column(JSON, nullable=True)
    # → {"communication": 0.7, "social": 0.6, "behavior": 0.5, "sensory": 0.4}

    # ── Fused risk output ───────────────────────────────────────
    final_risk_score = Column(Float, nullable=True)
    risk_level = Column(String(10), nullable=True)  # "low"/"medium"/"high"
    confidence = Column(Float, nullable=True)
    video_contribution = Column(String(20), nullable=True)
    questionnaire_contribution = Column(String(20), nullable=True)

    # ── Status tracking ─────────────────────────────────────────
    status = Column(String(25), nullable=False, default="pending", index=True)
    # Values: pending | video_uploaded | video_processing | video_done
    #         | questionnaire_done | complete | error_video | error_questionnaire

    # ── Timestamps ──────────────────────────────────────────────
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        sid = str(self.session_uuid)[:8] if self.session_uuid else "?"
        return f"<Session {sid}*** status={self.status}>"
