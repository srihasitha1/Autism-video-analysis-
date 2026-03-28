"""
tests/test_questionnaire.py
============================
Tests for Sprint 5 — Questionnaire Scoring API.

Covers:
  - GET  /api/v1/questionnaire/questions  (question list endpoint)
  - POST /api/v1/analyze/questionnaire    (scoring endpoint)
  - Input validation (wrong count, out-of-range, missing fields)
  - Scoring logic (all-zero, all-max, mixed)
  - Session status transitions
  - Unit tests for scoring internals
  - Schema validation

All tests mock the RF model to avoid needing the actual .pkl file.
"""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


async def _create_session(client) -> str:
    """Create a guest session via the API."""
    resp = await client.post("/api/v1/auth/guest")
    assert resp.status_code == 200, f"Guest session creation failed: {resp.text}"
    return resp.json()["session_uuid"]


async def _set_session_status(client, session_uuid: str, new_status: str, **extra):
    """Set session status directly in the test DB."""
    from app.db.session import get_db
    from app.main import app
    from app.models.session import AssessmentSession
    from sqlalchemy import select

    override_fn = app.dependency_overrides.get(get_db)
    if override_fn:
        async for db in override_fn():
            result = await db.execute(
                select(AssessmentSession).where(
                    AssessmentSession.session_uuid == uuid.UUID(session_uuid)
                )
            )
            session = result.scalar_one_or_none()
            if session:
                session.status = new_status
                for key, value in extra.items():
                    setattr(session, key, value)
                await db.commit()
            break


def _valid_payload(session_uuid: str) -> dict:
    """Build a valid questionnaire request payload."""
    return {
        "session_uuid": session_uuid,
        "child_age_months": 36,
        "child_gender": "male",
        "responses": [2] * 40,  # All "Sometimes"
    }


def _mock_rf_model():
    """Create a mock RF model that returns deterministic probabilities."""
    mock_model = MagicMock()
    # predict_proba returns [[prob_class_0, prob_class_1]]
    mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
    return mock_model


# ═══════════════════════════════════════════════════════════════
# GET /api/v1/questionnaire/questions
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_questions_returns_40(test_app):
    """GET /questions returns exactly 40 questions in 4 sections."""
    resp = await test_app.get("/api/v1/questionnaire/questions")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_questions"] == 40
    assert len(data["questions"]) == 40
    assert len(data["sections"]) == 4


@pytest.mark.asyncio
async def test_get_questions_has_scale(test_app):
    """GET /questions includes scale labels."""
    resp = await test_app.get("/api/v1/questionnaire/questions")
    data = resp.json()

    scale = data["scale"]
    assert scale["0"] == "Never"
    assert scale["4"] == "Always"


@pytest.mark.asyncio
async def test_get_questions_has_disclaimer(test_app):
    """GET /questions includes medical disclaimer."""
    resp = await test_app.get("/api/v1/questionnaire/questions")
    data = resp.json()

    assert "not a diagnostic" in data["disclaimer"].lower()


@pytest.mark.asyncio
async def test_get_questions_section_structure(test_app):
    """Each section has correct metadata."""
    resp = await test_app.get("/api/v1/questionnaire/questions")
    data = resp.json()

    sections = data["sections"]
    assert sections[0]["name"] == "Social Interaction"
    assert sections[0]["question_count"] == 10
    assert sections[1]["name"] == "Communication"
    assert sections[2]["name"] == "Behavior Patterns"
    assert sections[3]["name"] == "Sensory & Emotional"


# ═══════════════════════════════════════════════════════════════
# POST /api/v1/analyze/questionnaire — Validation
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_score_wrong_response_count_39(test_app):
    """Submitting 39 responses → 422."""
    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["responses"] = [2] * 39  # Too few

    resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_score_wrong_response_count_41(test_app):
    """Submitting 41 responses → 422."""
    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["responses"] = [2] * 41  # Too many

    resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_score_out_of_range_negative(test_app):
    """Response value -1 → 422."""
    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["responses"][0] = -1

    resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_score_out_of_range_5(test_app):
    """Response value 5 → 422."""
    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["responses"][0] = 5

    resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_score_invalid_age_too_low(test_app):
    """Age 0 months → 422."""
    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["child_age_months"] = 0

    resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_score_invalid_age_too_high(test_app):
    """Age 200 months → 422."""
    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["child_age_months"] = 200

    resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_score_invalid_gender(test_app):
    """Invalid gender value → 422."""
    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["child_gender"] = "robot"

    resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_score_session_not_found(test_app):
    """Non-existent session UUID → 404."""
    payload = _valid_payload(str(uuid.uuid4()))

    resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════════
# POST /api/v1/analyze/questionnaire — Scoring (Mocked Model)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_score_success_with_mock_model(test_app):
    """Valid submission with mocked RF model → 200 with results."""
    from app.services import questionnaire_scorer

    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)

    mock_model = _mock_rf_model()

    # Patch the model loading to return our mock
    questionnaire_scorer.reset_model_cache()
    questionnaire_scorer._rf_model = mock_model
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = True

    try:
        resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_uuid"] == session_uuid
        assert data["status"] == "questionnaire_done"
        assert "questionnaire_probability" in data
        assert "category_scores" in data
        assert "risk_level" in data
        assert data["risk_level"] in ("Low", "Moderate", "Elevated", "High")
        assert "disclaimer" in data
        assert "not a diagnostic" in data["disclaimer"].lower()
    finally:
        questionnaire_scorer.reset_model_cache()


@pytest.mark.asyncio
async def test_score_all_zeros_low_risk(test_app):
    """All-zero responses (Q1–Q20 inverted to 4) → some risk from inversion."""
    from app.services import questionnaire_scorer

    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["responses"] = [0] * 40

    # Mock model returns low probability
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

    questionnaire_scorer.reset_model_cache()
    questionnaire_scorer._rf_model = mock_model
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = True

    try:
        resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # All-zero with inversion → Q1–Q20 become 4 (high risk), Q21–Q40 stay 0
        # So social (=4*10/40=1.0) + comm (=1.0) but behavior (=0) + sensory (=0)
        # Weighted: 0.25*1.0 + 0.25*1.0 + 0.30*0 + 0.20*0 = 0.50
        # Hybrid: 0.6*0.2 + 0.4*0.50 = 0.12 + 0.20 = 0.32
        assert 0.0 <= data["questionnaire_probability"] <= 1.0
        assert data["category_scores"]["social_interaction"] == 1.0
        assert data["category_scores"]["behavior_patterns"] == 0.0
    finally:
        questionnaire_scorer.reset_model_cache()


@pytest.mark.asyncio
async def test_score_all_fours_high_risk(test_app):
    """All-4 responses → Q21–Q40 at max, Q1–Q20 inverted to 0."""
    from app.services import questionnaire_scorer

    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["responses"] = [4] * 40

    # Mock model returns high probability
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

    questionnaire_scorer.reset_model_cache()
    questionnaire_scorer._rf_model = mock_model
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = True

    try:
        resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # All-4 with inversion → Q1–Q20 become 0, Q21–Q40 stay 4
        # Social (=0), Comm (=0), Behavior (=4*10/40=1.0), Sensory (=1.0)
        # Weighted: 0.25*0 + 0.25*0 + 0.30*1.0 + 0.20*1.0 = 0.50
        # Hybrid: 0.6*0.9 + 0.4*0.50 = 0.54 + 0.20 = 0.74
        assert data["category_scores"]["social_interaction"] == 0.0
        assert data["category_scores"]["behavior_patterns"] == 1.0
    finally:
        questionnaire_scorer.reset_model_cache()


@pytest.mark.asyncio
async def test_score_without_model_fallback(test_app):
    """When RF model is not available, uses weighted-only scoring."""
    from app.services import questionnaire_scorer

    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)

    # Ensure no model is loaded
    questionnaire_scorer.reset_model_cache()
    questionnaire_scorer._rf_model = None
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = False

    try:
        resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Weighted-only scoring should still return valid results
        assert 0.0 <= data["questionnaire_probability"] <= 1.0
        assert data["risk_level"] in ("Low", "Moderate", "Elevated", "High")
    finally:
        questionnaire_scorer.reset_model_cache()


@pytest.mark.asyncio
async def test_score_updates_session_status(test_app):
    """Scoring updates session status to 'questionnaire_done'."""
    from app.services import questionnaire_scorer

    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)

    questionnaire_scorer.reset_model_cache()
    questionnaire_scorer._rf_model = None
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = False

    try:
        resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
        assert resp.status_code == 200

        # Verify session status via the status endpoint
        status_resp = await test_app.get(f"/api/v1/session/{session_uuid}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "questionnaire_done"
    finally:
        questionnaire_scorer.reset_model_cache()


@pytest.mark.asyncio
async def test_score_stores_child_info(test_app):
    """Scoring stores child_age_months and child_gender in session."""
    from app.db.session import get_db
    from app.main import app
    from app.models.session import AssessmentSession
    from app.services import questionnaire_scorer
    from sqlalchemy import select

    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    payload["child_age_months"] = 48
    payload["child_gender"] = "female"

    questionnaire_scorer.reset_model_cache()
    questionnaire_scorer._rf_model = None
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = False

    try:
        resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
        assert resp.status_code == 200

        # Read directly from test DB
        override_fn = app.dependency_overrides.get(get_db)
        async for db in override_fn():
            result = await db.execute(
                select(AssessmentSession).where(
                    AssessmentSession.session_uuid == uuid.UUID(session_uuid)
                )
            )
            session = result.scalar_one_or_none()
            assert session is not None
            assert session.child_age_months == 48
            assert session.child_gender == "female"
            assert session.questionnaire_probability is not None
            assert session.category_scores is not None
            break
    finally:
        questionnaire_scorer.reset_model_cache()


@pytest.mark.asyncio
async def test_score_idempotent_resubmission(test_app):
    """Re-submitting questionnaire overwrites previous results."""
    from app.services import questionnaire_scorer

    session_uuid = await _create_session(test_app)

    questionnaire_scorer.reset_model_cache()
    questionnaire_scorer._rf_model = None
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = False

    try:
        # First submission — all 1s
        # After inversion: Q1–Q20 become 3, Q21–Q40 stay 1
        # social=30/40=0.75, comm=0.75, behavior=10/40=0.25, sensory=0.25
        # weighted = 0.25*0.75 + 0.25*0.75 + 0.30*0.25 + 0.20*0.25 = 0.5
        payload1 = _valid_payload(session_uuid)
        payload1["responses"] = [1] * 40
        resp1 = await test_app.post("/api/v1/analyze/questionnaire", json=payload1)
        assert resp1.status_code == 200
        prob1 = resp1.json()["questionnaire_probability"]

        # Second submission — low social/comm risk, high behavior/sensory risk
        # Q1–Q20 = 4 (inverted to 0 → low risk), Q21–Q40 = 4 (high risk)
        # social=0, comm=0, behavior=1.0, sensory=1.0
        # weighted = 0.25*0 + 0.25*0 + 0.30*1.0 + 0.20*1.0 = 0.50
        # Hmm, still 0.5. Use asymmetric: Q1-20=0 (inv→4), Q21-40=4
        payload2 = _valid_payload(session_uuid)
        payload2["responses"] = [0] * 20 + [4] * 20
        resp2 = await test_app.post("/api/v1/analyze/questionnaire", json=payload2)
        assert resp2.status_code == 200
        prob2 = resp2.json()["questionnaire_probability"]

        # [0]*20+[4]*20: social=1.0, comm=1.0, behavior=1.0, sensory=1.0
        # weighted = 0.25+0.25+0.30+0.20 = 1.0  ≠ 0.5
        assert prob1 != prob2
    finally:
        questionnaire_scorer.reset_model_cache()


@pytest.mark.asyncio
async def test_score_category_scores_present(test_app):
    """Response includes all 4 category scores."""
    from app.services import questionnaire_scorer

    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)

    questionnaire_scorer.reset_model_cache()
    questionnaire_scorer._rf_model = None
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = False

    try:
        resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
        assert resp.status_code == 200
        scores = resp.json()["category_scores"]

        assert "social_interaction" in scores
        assert "communication" in scores
        assert "behavior_patterns" in scores
        assert "sensory_emotional" in scores

        # All between 0 and 1
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"
    finally:
        questionnaire_scorer.reset_model_cache()


@pytest.mark.asyncio
async def test_score_optional_gender(test_app):
    """Gender can be omitted (defaults to 'unspecified')."""
    from app.services import questionnaire_scorer

    session_uuid = await _create_session(test_app)
    payload = _valid_payload(session_uuid)
    del payload["child_gender"]

    questionnaire_scorer.reset_model_cache()
    questionnaire_scorer._rf_model = None
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = False

    try:
        resp = await test_app.post("/api/v1/analyze/questionnaire", json=payload)
        assert resp.status_code == 200
    finally:
        questionnaire_scorer.reset_model_cache()


# ═══════════════════════════════════════════════════════════════
# Unit Tests — Scoring Internals
# ═══════════════════════════════════════════════════════════════


def test_invert_responses():
    """Q1–Q20 scores are inverted (4 - r), Q21–Q40 unchanged."""
    from app.services.questionnaire_scorer import _invert_responses

    # All 2s → Q1–Q20 become 2, Q21–Q40 stay 2
    responses = [2] * 40
    inverted = _invert_responses(responses)
    assert inverted == [2] * 40  # 4-2 == 2, so no visible change

    # All 0s → Q1–Q20 become 4, Q21–Q40 stay 0
    responses = [0] * 40
    inverted = _invert_responses(responses)
    assert inverted[:20] == [4] * 20
    assert inverted[20:] == [0] * 20

    # All 4s → Q1–Q20 become 0, Q21–Q40 stay 4
    responses = [4] * 40
    inverted = _invert_responses(responses)
    assert inverted[:20] == [0] * 20
    assert inverted[20:] == [4] * 20


def test_classify_risk_boundaries():
    """Risk classification at each threshold boundary."""
    from app.services.questionnaire_scorer import _classify_risk

    assert _classify_risk(0.0) == "Low"
    assert _classify_risk(0.29) == "Low"
    assert _classify_risk(0.3) == "Moderate"
    assert _classify_risk(0.59) == "Moderate"
    assert _classify_risk(0.6) == "Elevated"
    assert _classify_risk(0.79) == "Elevated"
    assert _classify_risk(0.8) == "High"
    assert _classify_risk(1.0) == "High"


def test_encode_gender():
    """Gender encoding matches training data convention."""
    from app.services.questionnaire_scorer import _encode_gender

    assert _encode_gender("male") == 1
    assert _encode_gender("female") == 0
    assert _encode_gender("unspecified") == 0
    assert _encode_gender(None) == 0


def test_compute_category_scores():
    """Section scores computed correctly from inverted responses."""
    from app.services.questionnaire_scorer import _compute_category_scores

    # All responses = 2 → each section sum = 20, score = 20/40 = 0.5
    inverted = [2] * 40
    scores = _compute_category_scores(inverted)
    assert scores["social_interaction"] == 0.5
    assert scores["communication"] == 0.5
    assert scores["behavior_patterns"] == 0.5
    assert scores["sensory_emotional"] == 0.5


def test_score_questionnaire_deterministic():
    """Same inputs always produce same output (no randomness)."""
    from app.services.questionnaire_scorer import (
        reset_model_cache,
        score_questionnaire,
    )

    reset_model_cache()
    # Force no-model fallback
    from app.services import questionnaire_scorer
    questionnaire_scorer._rf_model = None
    questionnaire_scorer._model_loaded = True
    questionnaire_scorer._model_available = False

    try:
        r1 = score_questionnaire([2] * 40, 36, "male")
        r2 = score_questionnaire([2] * 40, 36, "male")
        assert r1["probability"] == r2["probability"]
        assert r1["category_scores"] == r2["category_scores"]
        assert r1["risk_level"] == r2["risk_level"]
    finally:
        reset_model_cache()


# ═══════════════════════════════════════════════════════════════
# Schema Validation Tests
# ═══════════════════════════════════════════════════════════════


def test_questionnaire_response_schema():
    """QuestionnaireResponse always includes disclaimer."""
    from app.schemas.questionnaire import QuestionnaireResponse

    resp = QuestionnaireResponse(
        session_uuid=uuid.uuid4(),
        questionnaire_probability=0.45,
        category_scores={"social_interaction": 0.5, "communication": 0.5},
        risk_level="Moderate",
        status="questionnaire_done",
    )

    assert resp.disclaimer
    assert "not a diagnostic" in resp.disclaimer.lower()


def test_questionnaire_request_rejects_empty_responses():
    """Empty responses list → validation error."""
    from pydantic import ValidationError

    from app.schemas.questionnaire import QuestionnaireRequest

    with pytest.raises(ValidationError):
        QuestionnaireRequest(
            session_uuid=uuid.uuid4(),
            child_age_months=36,
            responses=[],
        )


def test_questionnaire_request_rejects_non_int():
    """Non-integer response → validation error."""
    from pydantic import ValidationError

    from app.schemas.questionnaire import QuestionnaireRequest

    with pytest.raises(ValidationError):
        QuestionnaireRequest(
            session_uuid=uuid.uuid4(),
            child_age_months=36,
            responses=[2] * 39 + ["abc"],
        )
