"""
app/schemas/video.py
====================
Pydantic schemas for video analysis endpoints.

Covers upload (Sprint 3), start/status (Sprint 4).
"""

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


# ── Sprint 3: Upload ────────────────────────────────────────────

class VideoUploadResponse(BaseModel):
    """Response after successful video upload."""
    session_uuid: UUID
    status: str
    file_size_mb: float
    message: str = "Video uploaded successfully. Ready for analysis."


class VideoDeleteResponse(BaseModel):
    """Response after video temp files deleted."""
    session_uuid: UUID
    cleaned: bool
    message: str


# ── Sprint 4: Start & Status ───────────────────────────────────

class VideoStartRequest(BaseModel):
    """Request to start video analysis."""
    session_uuid: UUID


class VideoStartResponse(BaseModel):
    """Response after dispatching video analysis task."""
    session_uuid: UUID
    task_id: str
    status: str
    message: str = "Video analysis started. Poll /status for results."


class VideoStatusResponse(BaseModel):
    """Response for video analysis status polling."""
    session_uuid: UUID
    status: str
    task_id: Optional[str] = None
    task_state: Optional[str] = None

    # Results (populated when status == "video_done")
    video_class_probabilities: Optional[dict[str, Any]] = None
    video_confidence: Optional[str] = None
    video_score: Optional[float] = None
    risk_level: Optional[str] = None

    # Error info (populated when status == "error_video")
    error: Optional[str] = None

    # Always present
    disclaimer: str = (
        "This is an early behavioral risk assessment tool, not a diagnostic instrument. "
        "Please consult a qualified healthcare professional for clinical evaluation."
    )
