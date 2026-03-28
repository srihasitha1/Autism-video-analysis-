"""
tests/test_fusion.py
=====================
Tests for Sprint 6 — Fusion Engine & Risk Report API.

Covers:
  - Unit tests for pure fusion logic (`fusion_engine.py`)
  - POST /api/v1/analyze/fuse (normal, fallback, idempotent)
  - GET  /api/v1/analyze/report/{uuid} (gated by complete)
  - GET  /api/v1/analyze/report/{uuid}/summary
"""

import uuid

import pytest


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


async def _create_session(client) -> str:
    """Create a guest session via the API."""
    resp = await client.post("/api/v1/auth/guest")
    assert resp.status_code == 200, f"Guest session creation failed: {resp.text}"
    return resp.json()["session_uuid"]


async def _setup_session_for_fusion(client, session_uuid: str, has_video: bool = True):
    """Bypass the actual ML logic and inject raw scores into the DB."""
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
                session.questionnaire_probability = 0.8
                session.child_age_months = 36  # Not young (<24)
                session.status = "questionnaire_done"
                
                if has_video:
                    session.video_score = 0.6
                    session.video_confidence = "high"
                
                await db.commit()
            break


# ═══════════════════════════════════════════════════════════════
# Unit Tests — Fusion Engine (Pure Logic)
# ═══════════════════════════════════════════════════════════════


def test_fusion_high_confidence():
    """Video confidence 'high' implies weights 0.6/0.4."""
    from app.services.fusion_engine import fuse

    res = fuse(
        questionnaire_probability=0.8,
        video_prob=0.6,
        video_confidence="high",
        child_age_months=36,
    )
    assert res["weights_used"] == {"video": 0.6, "questionnaire": 0.4}
    # 0.6*0.6 + 0.4*0.8 = 0.36 + 0.32 = 0.68
    assert res["final_risk_score"] == 0.68
    assert res["risk_level"] == "medium"
    assert res["video_fallback_used"] is False


def test_fusion_medium_confidence():
    """Video confidence 'medium' implies weights 0.5/0.5."""
    from app.services.fusion_engine import fuse

    res = fuse(
        questionnaire_probability=0.8,
        video_prob=0.6,
        video_confidence="medium",
        child_age_months=36,
    )
    assert res["weights_used"] == {"video": 0.5, "questionnaire": 0.5}
    # 0.5*0.6 + 0.5*0.8 = 0.30 + 0.40 = 0.70
    assert res["final_risk_score"] == 0.70
    assert res["risk_level"] == "medium"


def test_fusion_low_confidence():
    """Video confidence 'low' implies weights 0.3/0.7."""
    from app.services.fusion_engine import fuse

    res = fuse(
        questionnaire_probability=0.8,
        video_prob=0.6,
        video_confidence="low",
        child_age_months=36,
    )
    assert res["weights_used"] == {"video": 0.3, "questionnaire": 0.7}
    # 0.3*0.6 + 0.7*0.8 = 0.18 + 0.56 = 0.74
    assert res["final_risk_score"] == 0.74
    assert res["risk_level"] == "high"


def test_fusion_age_adjustment():
    """Age < 24 months shifts weight +10% to questionnaire."""
    from app.services.fusion_engine import fuse

    res = fuse(
        questionnaire_probability=0.8,
        video_prob=0.6,
        video_confidence="high",
        child_age_months=18,  # < 24
    )
    # High base is 0.6/0.4. Age shifts it to 0.5/0.5.
    assert res["weights_used"] == {"video": 0.5, "questionnaire": 0.5}


def test_fusion_fallback_missing_video():
    """Missing video treats as 0.5 prob and low confidence."""
    from app.services.fusion_engine import fuse

    res = fuse(
        questionnaire_probability=0.8,
        video_prob=None,
        video_confidence=None,
        child_age_months=36,
    )
    # Low confidence -> 0.3/0.7
    assert res["weights_used"] == {"video": 0.3, "questionnaire": 0.7}
    assert res["video_fallback_used"] is True
    # 0.3*0.5 (fallback) + 0.7*0.8 = 0.15 + 0.56 = 0.71
    assert res["final_risk_score"] == 0.71


def test_fusion_boundaries():
    """Test the exact risk thresholds (0.35 and 0.70)."""
    from app.services.fusion_engine import _classify_risk

    assert _classify_risk(0.34) == "low"
    assert _classify_risk(0.35) == "medium"
    assert _classify_risk(0.70) == "medium"
    assert _classify_risk(0.71) == "high"


def test_fusion_confidence_formula():
    """Confidence = abs(score - 0.5) * 2."""
    from app.services.fusion_engine import fuse

    # Final score = 0.5
    res1 = fuse(questionnaire_probability=0.5, video_prob=0.5, video_confidence="medium")
    assert res1["confidence"] == 0.0  # abs(0.5 - 0.5) * 2

    # Final score = 1.0 (assuming weights 0.5/0.5)
    res2 = fuse(questionnaire_probability=1.0, video_prob=1.0, video_confidence="medium")
    assert res2["confidence"] == 1.0  # abs(1.0 - 0.5) * 2 = 1.0

    # Final score = 0.0
    res3 = fuse(questionnaire_probability=0.0, video_prob=0.0, video_confidence="medium")
    assert res3["confidence"] == 1.0  # abs(0.0 - 0.5) * 2 = 1.0


def test_fusion_clamp():
    """Ensure final output is clamped [0, 1] even if bad data provided."""
    from app.services.fusion_engine import fuse

    res = fuse(questionnaire_probability=2.0, video_prob=1.5, video_confidence="high")
    assert res["final_risk_score"] == 1.0


def test_fusion_contribution_labels():
    """Test standard concern labels per modality."""
    from app.services.fusion_engine import _classify_contribution

    assert _classify_contribution(0.60) == "high concern"
    assert _classify_contribution(0.35) == "moderate concern"
    assert _classify_contribution(0.34) == "low concern"


# ═══════════════════════════════════════════════════════════════
# Integration Tests — POST /fuse
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_fuse_missing_session(test_app):
    """404 if session doesn't exist."""
    resp = await test_app.post("/api/v1/analyze/fuse", json={"session_uuid": str(uuid.uuid4())})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_fuse_missing_questionnaire(test_app):
    """409 if questionnaire has not been completed."""
    session_uuid = await _create_session(test_app)
    # Session is fresh, so questionnaire_probability is None
    resp = await test_app.post("/api/v1/analyze/fuse", json={"session_uuid": session_uuid})
    assert resp.status_code == 409
    assert "Questionnaire must be completed" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_fuse_success_with_video(test_app):
    """200 when both questionnaire and video exist."""
    session_uuid = await _create_session(test_app)
    await _setup_session_for_fusion(test_app, session_uuid, has_video=True)

    resp = await test_app.post("/api/v1/analyze/fuse", json={"session_uuid": session_uuid})
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_uuid"] == session_uuid
    assert data["status"] == "complete"
    assert data["video_fallback_used"] is False
    assert "final_risk_score" in data
    assert "risk_level" in data
    assert data["risk_level"] in ("low", "medium", "high")


@pytest.mark.asyncio
async def test_fuse_success_fallback_video(test_app):
    """200 using fallback when video is missing but questionnaire is done."""
    session_uuid = await _create_session(test_app)
    await _setup_session_for_fusion(test_app, session_uuid, has_video=False)

    resp = await test_app.post("/api/v1/analyze/fuse", json={"session_uuid": session_uuid})
    assert resp.status_code == 200
    data = resp.json()

    assert data["video_fallback_used"] is True
    assert data["status"] == "complete"


@pytest.mark.asyncio
async def test_fuse_idempotent(test_app):
    """Calling fuse twice overwrites with same result."""
    session_uuid = await _create_session(test_app)
    await _setup_session_for_fusion(test_app, session_uuid, has_video=True)

    resp1 = await test_app.post("/api/v1/analyze/fuse", json={"session_uuid": session_uuid})
    assert resp1.status_code == 200

    resp2 = await test_app.post("/api/v1/analyze/fuse", json={"session_uuid": session_uuid})
    assert resp2.status_code == 200

    assert resp1.json() == resp2.json()


# ═══════════════════════════════════════════════════════════════
# Integration Tests — GET /report
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_report_not_found(test_app):
    """404 if session doesn't exist."""
    resp = await test_app.get(f"/api/v1/analyze/report/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_report_not_complete(test_app):
    """409 if session is not complete yet."""
    session_uuid = await _create_session(test_app)
    # Session is fresh, status="pending"
    resp = await test_app.get(f"/api/v1/analyze/report/{session_uuid}")
    assert resp.status_code == 409
    assert "not ready" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_report_full_success(test_app):
    """200 returns full report after complete."""
    session_uuid = await _create_session(test_app)
    await _setup_session_for_fusion(test_app, session_uuid, has_video=True)
    await test_app.post("/api/v1/analyze/fuse", json={"session_uuid": session_uuid})

    resp = await test_app.get(f"/api/v1/analyze/report/{session_uuid}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_uuid"] == session_uuid
    assert data["status"] == "complete"
    assert "final_risk_score" in data
    assert "video_score" in data
    assert "questionnaire_probability" in data
    assert "weights_used" in data
    assert "disclaimer" in data


@pytest.mark.asyncio
async def test_report_summary_success(test_app):
    """200 returns lightweight summary after complete."""
    session_uuid = await _create_session(test_app)
    await _setup_session_for_fusion(test_app, session_uuid, has_video=True)
    await test_app.post("/api/v1/analyze/fuse", json={"session_uuid": session_uuid})

    resp = await test_app.get(f"/api/v1/analyze/report/{session_uuid}/summary")
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_uuid"] == session_uuid
    assert data["status"] == "complete"
    assert "final_risk_score" in data
    assert "risk_level" in data
    # Summary should not contain detailed fields
    assert "video_score" not in data
    assert "weights_used" not in data
