"""
app/schemas/questionnaire.py
=============================
Pydantic schemas for the questionnaire scoring endpoints.

Covers:
  - GET  /api/v1/questionnaire/questions  → QuestionsListResponse
  - POST /api/v1/analyze/questionnaire    → QuestionnaireResponse
"""

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ─────────────────────────────────────────────


class QuestionnaireRequest(BaseModel):
    """Request body for questionnaire scoring."""

    session_uuid: UUID
    child_age_months: int = Field(
        ...,
        ge=12,
        le=96,
        description="Child's age in months (12–96, i.e. 1–8 years).",
    )
    child_gender: Optional[str] = Field(
        default="unspecified",
        description="Child's gender: 'male', 'female', or 'unspecified'.",
    )
    responses: list[int] = Field(
        ...,
        description="Exactly 40 integer responses, each in [0, 4].",
    )

    @field_validator("child_gender")
    @classmethod
    def validate_gender(cls, v: str | None) -> str:
        if v is None:
            return "unspecified"
        allowed = {"male", "female", "unspecified"}
        if v.lower() not in allowed:
            raise ValueError(
                f"child_gender must be one of {sorted(allowed)}, got '{v}'"
            )
        return v.lower()

    @field_validator("responses")
    @classmethod
    def validate_responses(cls, v: list[int]) -> list[int]:
        if len(v) != 40:
            raise ValueError(
                f"responses must contain exactly 40 items, got {len(v)}"
            )
        for i, val in enumerate(v):
            if not isinstance(val, int) or val < 0 or val > 4:
                raise ValueError(
                    f"Response at index {i} must be an integer between 0 and 4, got {val}"
                )
        return v


# ── Response schemas ────────────────────────────────────────────


class QuestionnaireResponse(BaseModel):
    """Response after questionnaire scoring."""

    session_uuid: UUID
    questionnaire_probability: float
    category_scores: dict[str, float]
    risk_level: str
    status: str
    disclaimer: str = (
        "This is an early behavioral risk assessment tool, not a diagnostic instrument. "
        "Please consult a qualified healthcare professional for clinical evaluation."
    )


class QuestionItemSchema(BaseModel):
    """A single question for the frontend."""

    id: int
    text: str
    section: str
    section_index: int


class SectionSchema(BaseModel):
    """Section metadata for the frontend."""

    name: str
    index: int
    question_count: int


class QuestionsListResponse(BaseModel):
    """Response for the GET questions endpoint."""

    sections: list[SectionSchema]
    questions: list[QuestionItemSchema]
    scale: dict[int, str]
    total_questions: int
    disclaimer: str = (
        "This is an early behavioral risk assessment tool, not a diagnostic instrument. "
        "Please consult a qualified healthcare professional for clinical evaluation."
    )
