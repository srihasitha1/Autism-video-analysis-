"""
tests/test_video.py
===================
Sprint 3 verification: video upload, validation, storage, and cleanup.
"""

import io
import os
import uuid
from pathlib import Path
from unittest.mock import patch

# ── Helpers: build fake video files ─────────────────────────────

def _mp4_header() -> bytes:
    """Minimal MP4 magic bytes: 4 bytes size + 'ftyp' + 'isom'."""
    return b"\x00\x00\x00\x18ftypMSNV" + b"\x00" * 100


def _avi_header() -> bytes:
    """Minimal AVI magic bytes: RIFF header."""
    return b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 100


def _webm_header() -> bytes:
    """EBML header for WebM."""
    return b"\x1a\x45\xdf\xa3" + b"\x00" * 100


def _fake_video_file(header: bytes = None, size: int = 1024, ext: str = "mp4"):
    """Create a BytesIO that simulates a video upload."""
    if header is None:
        header = _mp4_header()
    content = header + b"\x00" * max(0, size - len(header))
    return ("test_video." + ext, content, "video/mp4")


async def _create_session(client) -> str:
    """Helper: create a guest session and return its UUID."""
    resp = await client.post("/api/v1/auth/guest")
    return resp.json()["session_uuid"]


# ════════════════════════════════════════════════════════════════
# Upload tests
# ════════════════════════════════════════════════════════════════

async def test_upload_valid_mp4(test_app, tmp_path):
    """Upload a valid MP4 file — should succeed with 200."""
    session_uuid = await _create_session(test_app)
    header = _mp4_header()
    content = header + b"\x00" * 500

    resp = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("test.mp4", io.BytesIO(content), "video/mp4")},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "video_uploaded"
    assert data["session_uuid"] == session_uuid
    assert data["file_size_mb"] >= 0


async def test_upload_valid_avi(test_app):
    """Upload a valid AVI file — should succeed."""
    session_uuid = await _create_session(test_app)
    content = _avi_header()

    resp = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("test.avi", io.BytesIO(content), "video/avi")},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "video_uploaded"


async def test_upload_valid_webm(test_app):
    """Upload a valid WebM file — should succeed."""
    session_uuid = await _create_session(test_app)
    content = _webm_header()

    resp = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("test.webm", io.BytesIO(content), "video/webm")},
    )

    assert resp.status_code == 200


# ════════════════════════════════════════════════════════════════
# Extension validation tests
# ════════════════════════════════════════════════════════════════

async def test_upload_bad_extension(test_app):
    """Reject a .exe file even with valid video magic bytes."""
    session_uuid = await _create_session(test_app)
    content = _mp4_header()

    resp = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("malware.exe", io.BytesIO(content), "application/octet-stream")},
    )

    assert resp.status_code == 400
    assert "not allowed" in resp.json()["detail"].lower()


async def test_upload_no_extension(test_app):
    """Reject a file with no extension."""
    session_uuid = await _create_session(test_app)
    content = _mp4_header()

    resp = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("video_no_ext", io.BytesIO(content), "video/mp4")},
    )

    assert resp.status_code == 400


# ════════════════════════════════════════════════════════════════
# Magic bytes validation tests
# ════════════════════════════════════════════════════════════════

async def test_upload_bad_magic_bytes(test_app):
    """Reject a .mp4 file whose header is actually a PNG."""
    session_uuid = await _create_session(test_app)
    # PNG magic bytes disguised as .mp4
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    resp = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("fake_video.mp4", io.BytesIO(png_header), "video/mp4")},
    )

    assert resp.status_code == 400
    assert "does not appear to be a valid video" in resp.json()["detail"]


async def test_upload_too_small_file(test_app):
    """Reject a file that's too small to have valid magic bytes."""
    session_uuid = await _create_session(test_app)
    tiny = b"\x00" * 5  # Only 5 bytes

    resp = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("tiny.mp4", io.BytesIO(tiny), "video/mp4")},
    )

    assert resp.status_code == 400
    assert "too small" in resp.json()["detail"].lower()


# ════════════════════════════════════════════════════════════════
# Session validation tests
# ════════════════════════════════════════════════════════════════

async def test_upload_missing_session(test_app):
    """Reject upload with a nonexistent session UUID."""
    content = _mp4_header()
    fake_uuid = str(uuid.uuid4())

    resp = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": fake_uuid},
        files={"file": ("test.mp4", io.BytesIO(content), "video/mp4")},
    )

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


async def test_upload_invalid_uuid_format(test_app):
    """Reject upload with malformed UUID."""
    content = _mp4_header()

    resp = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": "not-a-uuid"},
        files={"file": ("test.mp4", io.BytesIO(content), "video/mp4")},
    )

    assert resp.status_code == 400
    assert "invalid" in resp.json()["detail"].lower()


# ════════════════════════════════════════════════════════════════
# Session status transitions
# ════════════════════════════════════════════════════════════════

async def test_upload_updates_session_status(test_app):
    """After successful upload, session status should be 'video_uploaded'."""
    session_uuid = await _create_session(test_app)
    content = _mp4_header()

    await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("test.mp4", io.BytesIO(content), "video/mp4")},
    )

    # Check session status via the GET endpoint
    status_resp = await test_app.get(f"/api/v1/session/{session_uuid}")
    assert status_resp.json()["status"] == "video_uploaded"


async def test_reupload_allowed(test_app):
    """Uploading again to a 'video_uploaded' session should succeed (re-upload)."""
    session_uuid = await _create_session(test_app)
    content = _mp4_header()

    # First upload
    resp1 = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("test1.mp4", io.BytesIO(content), "video/mp4")},
    )
    assert resp1.status_code == 200

    # Second upload (re-upload)
    resp2 = await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("test2.mp4", io.BytesIO(content), "video/mp4")},
    )
    assert resp2.status_code == 200


# ════════════════════════════════════════════════════════════════
# Deletion tests
# ════════════════════════════════════════════════════════════════

async def test_delete_video_success(test_app):
    """DELETE /api/v1/analyze/video/{uuid} should clean temp files."""
    session_uuid = await _create_session(test_app)
    content = _mp4_header()

    # Upload first
    await test_app.post(
        "/api/v1/analyze/video/upload",
        data={"session_uuid": session_uuid},
        files={"file": ("test.mp4", io.BytesIO(content), "video/mp4")},
    )

    # Delete
    del_resp = await test_app.delete(f"/api/v1/analyze/video/{session_uuid}")
    assert del_resp.status_code == 200
    data = del_resp.json()
    assert data["cleaned"] is True


async def test_delete_video_no_files(test_app):
    """Deleting when no temp files exist should return cleaned=False."""
    session_uuid = await _create_session(test_app)

    del_resp = await test_app.delete(f"/api/v1/analyze/video/{session_uuid}")
    assert del_resp.status_code == 200
    assert del_resp.json()["cleaned"] is False


async def test_delete_video_nonexistent_session(test_app):
    """DELETE with unknown session returns 404."""
    fake_uuid = str(uuid.uuid4())
    resp = await test_app.delete(f"/api/v1/analyze/video/{fake_uuid}")
    assert resp.status_code == 404


# ════════════════════════════════════════════════════════════════
# Unit tests for validator functions
# ════════════════════════════════════════════════════════════════

def test_validate_extension_valid():
    """Known extensions should pass."""
    from app.utils.validators import validate_extension

    assert validate_extension("video.mp4") == ".mp4"
    assert validate_extension("video.MP4") == ".mp4"  # case-insensitive
    assert validate_extension("video.mov") == ".mov"
    assert validate_extension("video.avi") == ".avi"
    assert validate_extension("video.webm") == ".webm"


def test_validate_extension_invalid():
    """Unknown extensions should raise ValueError."""
    import pytest
    from app.utils.validators import validate_extension

    with pytest.raises(ValueError, match="not allowed"):
        validate_extension("video.exe")

    with pytest.raises(ValueError, match="not allowed"):
        validate_extension("video.gif")

    with pytest.raises(ValueError):
        validate_extension("noextension")


def test_validate_magic_bytes_valid():
    """Known video headers should pass."""
    from app.utils.validators import validate_magic_bytes

    assert validate_magic_bytes(_mp4_header()) is True
    assert validate_magic_bytes(_avi_header()) is True
    assert validate_magic_bytes(_webm_header()) is True


def test_validate_magic_bytes_invalid():
    """Non-video headers should raise ValueError."""
    import pytest
    from app.utils.validators import validate_magic_bytes

    # PNG header
    with pytest.raises(ValueError, match="does not appear"):
        validate_magic_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

    # Random bytes
    with pytest.raises(ValueError, match="does not appear"):
        validate_magic_bytes(b"\xde\xad\xbe\xef" * 4)

    # Too short
    with pytest.raises(ValueError, match="too small"):
        validate_magic_bytes(b"\x00" * 5)


def test_validate_file_size():
    """Size validation should reject files over the limit."""
    import pytest
    from app.utils.validators import validate_file_size

    # Under limit — no error
    validate_file_size(10 * 1024 * 1024, max_mb=50)

    # None content length — no error (can't validate)
    validate_file_size(None, max_mb=50)

    # Over limit — should raise
    with pytest.raises(ValueError, match="too large"):
        validate_file_size(100 * 1024 * 1024, max_mb=50)


# ════════════════════════════════════════════════════════════════
# Unit tests for privacy utilities
# ════════════════════════════════════════════════════════════════

def test_anonymize_uuid():
    """UUID anonymization should show only first 8 chars."""
    from app.utils.privacy import anonymize_uuid

    test_uuid = "12345678-1234-1234-1234-123456789abc"
    assert anonymize_uuid(test_uuid) == "12345678***"


def test_cleanup_nonexistent_session():
    """Cleaning up a session that doesn't exist should return False."""
    from app.utils.privacy import cleanup_session_video

    result = cleanup_session_video(uuid.uuid4())
    assert result is False
