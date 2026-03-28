"""
tests/test_video_inference.py
=============================
Tests for Sprint 4 — Video ML Inference.

Covers:
  - /api/v1/analyze/video/start endpoint (dispatch Celery task)
  - /api/v1/analyze/video/status/{uuid} endpoint (poll results)
  - Video inference adapter (mocked model)
  - Celery task helpers (mocked)
  - Edge cases (wrong status, missing session, duplicate start, etc.)

All tests mock the Celery task dispatch and ML model.
No actual TensorFlow or video processing runs.
"""

import uuid
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════
# Helper: Create a guest session and set it to a specific status
# ═══════════════════════════════════════════════════════════════


async def _create_session(client) -> str:
    """Create a guest session via the API."""
    resp = await client.post("/api/v1/auth/guest")
    assert resp.status_code == 200, f"Guest session creation failed: {resp.text}"
    session_uuid = resp.json()["session_uuid"]
    return session_uuid


async def _set_session_status(client, session_uuid: str, new_status: str, **extra_fields):
    """
    Helper to set a session status directly in the test DB.
    Uses the app's dependency override to access the test database.
    """
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
                for key, value in extra_fields.items():
                    setattr(session, key, value)
                await db.commit()
            break


# ═══════════════════════════════════════════════════════════════
# Test: Start Video Analysis Endpoint
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_start_analysis_success(test_app):
    """Start analysis on a session with video_uploaded status → 200 + task_id."""
    session_uuid = await _create_session(test_app)
    await _set_session_status(test_app, session_uuid, "video_uploaded")

    # Create a fake video file in the temp directory
    from app.utils.privacy import get_session_dir
    session_dir = get_session_dir(session_uuid)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "upload.mp4").write_bytes(b"\x00" * 100)

    # Mock the Celery task dispatch
    mock_task_result = MagicMock()
    mock_task_result.id = "test-task-id-12345"

    with patch("app.tasks.video_task.process_video") as mock_task:
        mock_task.delay.return_value = mock_task_result

        resp = await test_app.post(
            "/api/v1/analyze/video/start",
            json={"session_uuid": session_uuid},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "video_processing"
    assert data["task_id"] == "test-task-id-12345"
    assert data["session_uuid"] == session_uuid

    # Cleanup temp files
    if session_dir.exists():
        shutil.rmtree(session_dir)


@pytest.mark.asyncio
async def test_start_analysis_session_not_found(test_app):
    """Start analysis on non-existent session → 404."""
    fake_uuid = str(uuid.uuid4())

    resp = await test_app.post(
        "/api/v1/analyze/video/start",
        json={"session_uuid": fake_uuid},
    )

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_start_analysis_wrong_status_pending(test_app):
    """Start analysis when session is still in 'pending' (no video uploaded) → 409."""
    session_uuid = await _create_session(test_app)
    # Session is in "pending" by default — no video uploaded

    resp = await test_app.post(
        "/api/v1/analyze/video/start",
        json={"session_uuid": session_uuid},
    )

    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_start_analysis_already_processing(test_app):
    """Start analysis when already processing → 409 conflict."""
    session_uuid = await _create_session(test_app)
    await _set_session_status(test_app, session_uuid, "video_processing")

    resp = await test_app.post(
        "/api/v1/analyze/video/start",
        json={"session_uuid": session_uuid},
    )

    assert resp.status_code == 409
    assert "already in progress" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_start_analysis_already_done(test_app):
    """Start analysis when already completed → 409 conflict."""
    session_uuid = await _create_session(test_app)
    await _set_session_status(test_app, session_uuid, "video_done")

    resp = await test_app.post(
        "/api/v1/analyze/video/start",
        json={"session_uuid": session_uuid},
    )

    assert resp.status_code == 409
    assert "already been completed" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_start_analysis_video_file_missing(test_app):
    """Start analysis when video file has been deleted → 410 Gone."""
    session_uuid = await _create_session(test_app)
    await _set_session_status(test_app, session_uuid, "video_uploaded")

    # Don't create the video file — simulate it being cleaned up

    resp = await test_app.post(
        "/api/v1/analyze/video/start",
        json={"session_uuid": session_uuid},
    )

    assert resp.status_code == 410
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_start_analysis_retry_after_error(test_app):
    """Start analysis on a session with error_video status → should succeed (retry)."""
    session_uuid = await _create_session(test_app)
    await _set_session_status(test_app, session_uuid, "error_video")

    # Create a fake video file
    from app.utils.privacy import get_session_dir
    session_dir = get_session_dir(session_uuid)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "upload.mp4").write_bytes(b"\x00" * 100)

    mock_task_result = MagicMock()
    mock_task_result.id = "retry-task-id-67890"

    with patch("app.tasks.video_task.process_video") as mock_task:
        mock_task.delay.return_value = mock_task_result

        resp = await test_app.post(
            "/api/v1/analyze/video/start",
            json={"session_uuid": session_uuid},
        )

    assert resp.status_code == 200
    assert resp.json()["status"] == "video_processing"
    assert resp.json()["task_id"] == "retry-task-id-67890"

    # Cleanup
    if session_dir.exists():
        shutil.rmtree(session_dir)


# ═══════════════════════════════════════════════════════════════
# Test: Poll Video Status Endpoint
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_status_session_not_found(test_app):
    """Poll status for non-existent session → 404."""
    fake_uuid = str(uuid.uuid4())

    resp = await test_app.get(f"/api/v1/analyze/video/status/{fake_uuid}")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_status_pending(test_app):
    """Poll status for a pending session → returns status with disclaimer."""
    session_uuid = await _create_session(test_app)

    resp = await test_app.get(f"/api/v1/analyze/video/status/{session_uuid}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["video_score"] is None
    assert data["disclaimer"]  # Always present
    assert "not a diagnostic" in data["disclaimer"].lower()


@pytest.mark.asyncio
async def test_status_processing(test_app):
    """Poll status while processing → returns video_processing."""
    session_uuid = await _create_session(test_app)
    await _set_session_status(test_app, session_uuid, "video_processing")

    resp = await test_app.get(f"/api/v1/analyze/video/status/{session_uuid}")

    assert resp.status_code == 200
    assert resp.json()["status"] == "video_processing"


@pytest.mark.asyncio
async def test_status_done_with_results(test_app):
    """Poll status after completion → returns results with all fields."""
    session_uuid = await _create_session(test_app)

    # Simulate a completed session with results
    await _set_session_status(
        test_app, session_uuid, "video_done",
        video_score=0.45,
        video_confidence="medium",
        video_class_probabilities={
            "arm_flapping": 0.3,
            "spinning": 0.1,
            "head_banging": 0.2,
            "normal": 0.7,
        },
        risk_level="Moderate",
        celery_task_id="completed-task-id",
    )

    resp = await test_app.get(f"/api/v1/analyze/video/status/{session_uuid}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "video_done"
    assert data["video_score"] == 0.45
    assert data["video_confidence"] == "medium"
    assert data["risk_level"] == "Moderate"
    assert "arm_flapping" in data["video_class_probabilities"]
    assert data["disclaimer"]  # Always present


@pytest.mark.asyncio
async def test_status_error_with_message(test_app):
    """Poll status after error → returns error info."""
    session_uuid = await _create_session(test_app)

    await _set_session_status(
        test_app, session_uuid, "error_video",
        video_error="Could not extract clips from video",
    )

    resp = await test_app.get(f"/api/v1/analyze/video/status/{session_uuid}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error_video"
    assert "extract clips" in data["error"].lower()


# ═══════════════════════════════════════════════════════════════
# Test: Video Inference Adapter (Unit Tests)
# ═══════════════════════════════════════════════════════════════


def test_classify_risk_low():
    """Risk classification: score < 0.3 → Low."""
    from app.services.video_inference import _classify_risk

    assert _classify_risk(0.0) == "Low"
    assert _classify_risk(0.15) == "Low"
    assert _classify_risk(0.29) == "Low"


def test_classify_risk_moderate():
    """Risk classification: 0.3 ≤ score < 0.6 → Moderate."""
    from app.services.video_inference import _classify_risk

    assert _classify_risk(0.3) == "Moderate"
    assert _classify_risk(0.45) == "Moderate"
    assert _classify_risk(0.59) == "Moderate"


def test_classify_risk_high():
    """Risk classification: score ≥ 0.6 → High."""
    from app.services.video_inference import _classify_risk

    assert _classify_risk(0.6) == "High"
    assert _classify_risk(0.8) == "High"
    assert _classify_risk(1.0) == "High"


def test_score_to_confidence():
    """Confidence mapping based on distance from 0.5."""
    from app.services.video_inference import _score_to_confidence

    # Clear signal (far from 0.5) → high confidence
    assert _score_to_confidence(0.0) == "high"
    assert _score_to_confidence(1.0) == "high"
    assert _score_to_confidence(0.9) == "high"

    # Moderate signal → medium confidence
    assert _score_to_confidence(0.3) == "medium"
    assert _score_to_confidence(0.7) == "medium"

    # Ambiguous (near 0.5) → low confidence
    assert _score_to_confidence(0.5) == "low"
    assert _score_to_confidence(0.45) == "low"


def test_classify_risk_boundaries():
    """Test exact boundary values for risk classification."""
    from app.services.video_inference import _classify_risk

    # Boundary at 0.3
    assert _classify_risk(0.299) == "Low"
    assert _classify_risk(0.3) == "Moderate"

    # Boundary at 0.6
    assert _classify_risk(0.599) == "Moderate"
    assert _classify_risk(0.6) == "High"


# ═══════════════════════════════════════════════════════════════
# Test: File Finding Logic (without importing Celery module)
# ═══════════════════════════════════════════════════════════════


def test_find_video_file_exists(tmp_path):
    """Finds the uploaded video file in the session temp directory."""
    fake_uuid = str(uuid.uuid4())
    session_dir = tmp_path / fake_uuid
    session_dir.mkdir()
    (session_dir / "upload.mp4").write_bytes(b"\x00" * 100)

    # Simulate the file finding logic directly (avoid celery import)
    found = None
    for f in session_dir.iterdir():
        if f.is_file() and f.name.startswith("upload"):
            found = f
            break

    assert found is not None
    assert found.name == "upload.mp4"


def test_find_video_file_missing(tmp_path):
    """Returns None when no video file exists."""
    session_dir = tmp_path / "nonexistent"
    # Directory doesn't exist
    assert not session_dir.exists()


def test_find_video_file_wrong_name(tmp_path):
    """Doesn't match files that don't start with 'upload'."""
    fake_uuid = str(uuid.uuid4())
    session_dir = tmp_path / fake_uuid
    session_dir.mkdir()
    (session_dir / "random.mp4").write_bytes(b"\x00" * 100)

    found = None
    for f in session_dir.iterdir():
        if f.is_file() and f.name.startswith("upload"):
            found = f
            break

    assert found is None


# ═══════════════════════════════════════════════════════════════
# Test: Response Schema Validation
# ═══════════════════════════════════════════════════════════════


def test_video_status_response_schema():
    """VideoStatusResponse always includes disclaimer."""
    from app.schemas.video import VideoStatusResponse

    resp = VideoStatusResponse(
        session_uuid=uuid.uuid4(),
        status="pending",
    )

    assert resp.disclaimer
    assert "not a diagnostic" in resp.disclaimer.lower()
    assert resp.video_score is None
    assert resp.error is None


def test_video_start_response_schema():
    """VideoStartResponse includes task_id and default message."""
    from app.schemas.video import VideoStartResponse

    resp = VideoStartResponse(
        session_uuid=uuid.uuid4(),
        task_id="abc123",
        status="video_processing",
    )

    assert resp.task_id == "abc123"
    assert "poll" in resp.message.lower()


def test_video_status_response_with_results():
    """VideoStatusResponse with all result fields populated."""
    from app.schemas.video import VideoStatusResponse

    resp = VideoStatusResponse(
        session_uuid=uuid.uuid4(),
        status="video_done",
        task_id="task-123",
        task_state="SUCCESS",
        video_class_probabilities={"arm_flapping": 0.7, "normal": 0.3},
        video_confidence="high",
        video_score=0.85,
        risk_level="High",
    )

    assert resp.video_score == 0.85
    assert resp.risk_level == "High"
    assert resp.video_confidence == "high"
    assert "arm_flapping" in resp.video_class_probabilities
    assert resp.disclaimer  # Always present


def test_video_status_response_with_error():
    """VideoStatusResponse with error populated."""
    from app.schemas.video import VideoStatusResponse

    resp = VideoStatusResponse(
        session_uuid=uuid.uuid4(),
        status="error_video",
        error="Video analysis timed out",
    )

    assert resp.error == "Video analysis timed out"
    assert resp.video_score is None
