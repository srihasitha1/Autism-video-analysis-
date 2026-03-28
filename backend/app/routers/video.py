"""
app/routers/video.py
====================
Video upload, analysis, and status endpoints.

Endpoints:
  POST   /api/v1/analyze/video/upload          — Upload a video for analysis (Sprint 3)
  DELETE /api/v1/analyze/video/{uuid}           — Manually delete temp video files (Sprint 3)
  POST   /api/v1/analyze/video/start            — Dispatch ML inference task (Sprint 4)
  GET    /api/v1/analyze/video/status/{uuid}    — Poll inference status (Sprint 4)
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.models.session import AssessmentSession
from app.schemas.video import (
    VideoDeleteResponse,
    VideoStartRequest,
    VideoStartResponse,
    VideoStatusResponse,
    VideoUploadResponse,
)
from app.utils.privacy import (
    anonymize_uuid,
    cleanup_session_video,
    get_session_dir,
    save_upload_to_temp,
)
from app.utils.validators import validate_extension, validate_file_size, validate_magic_bytes

logger = logging.getLogger("autisense.video")

router = APIRouter(prefix="/analyze/video", tags=["Video Analysis"])


# ═══════════════════════════════════════════════════════════════
# SPRINT 3 — Upload & Delete
# ═══════════════════════════════════════════════════════════════


# ── POST /api/v1/analyze/video/upload ──────────────────────────
@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    session_uuid: str = Form(..., description="Assessment session UUID"),
    file: UploadFile = File(..., description="Video file (.mp4, .mov, .avi, .webm)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a video file for autism behavior analysis.

    Validation pipeline:
    1. Session exists and is in valid status
    2. File extension is allowed
    3. Content-Length within limit
    4. Magic bytes match a known video container
    5. Streaming write with actual byte-count enforcement

    The video is saved to a per-session temp directory.
    It will be consumed by the async inference task (Sprint 4)
    and auto-deleted after processing.
    """
    # ── Parse UUID safely ───────────────────────────────────────
    try:
        parsed_uuid = UUID(session_uuid)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_uuid format",
        )

    # ── Validate session exists and is in valid state ───────────
    result = await db.execute(
        select(AssessmentSession).where(
            AssessmentSession.session_uuid == parsed_uuid
        )
    )
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Only allow upload if session is in an appropriate state
    valid_statuses = {"pending", "video_uploaded", "error_video"}
    if session.status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot upload video when session status is '{session.status}'. "
            f"Must be one of: {', '.join(sorted(valid_statuses))}",
        )

    # ── Validate file extension ─────────────────────────────────
    try:
        ext = validate_extension(file.filename or "")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # ── Validate declared Content-Length ─────────────────────────
    try:
        validate_file_size(file.size, settings.MAX_VIDEO_SIZE_MB)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        )

    # ── Validate magic bytes ────────────────────────────────────
    try:
        header = await file.read(12)
        validate_magic_bytes(header)
        await file.seek(0)  # Reset for streaming save
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # ── Clean up any previous upload for this session ───────────
    cleanup_session_video(parsed_uuid)

    # ── Stream to temp storage ──────────────────────────────────
    try:
        video_path = await save_upload_to_temp(
            file_obj=file,
            session_uuid=parsed_uuid,
            extension=ext,
            max_size_mb=settings.MAX_VIDEO_SIZE_MB,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Upload failed for session %s: %s",
            anonymize_uuid(parsed_uuid),
            type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Video upload failed. Please try again.",
        )

    # ── Update session status in DB ─────────────────────────────
    session.status = "video_uploaded"
    await db.commit()

    file_size_mb = video_path.stat().st_size / (1024 * 1024)

    logger.info(
        "Upload complete: session=%s size=%.1f MB",
        anonymize_uuid(parsed_uuid),
        file_size_mb,
    )

    return VideoUploadResponse(
        session_uuid=parsed_uuid,
        status="video_uploaded",
        file_size_mb=round(file_size_mb, 2),
    )


# ── DELETE /api/v1/analyze/video/{session_uuid} ────────────────
@router.delete("/{session_uuid}", response_model=VideoDeleteResponse)
async def delete_video(
    session_uuid: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger cleanup of temp video files for a session.
    This is a safety valve — files are also auto-deleted after inference
    and by the stale cleanup scheduler.
    """
    # Verify session exists
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

    # Clean up temp files
    cleaned = cleanup_session_video(session_uuid)

    # If session was in video_uploaded state, reset to pending
    if session.status == "video_uploaded":
        session.status = "pending"
        await db.commit()

    return VideoDeleteResponse(
        session_uuid=session_uuid,
        cleaned=cleaned,
        message="Temp video files deleted" if cleaned else "No temp files found",
    )


# ═══════════════════════════════════════════════════════════════
# SPRINT 4 — Start Inference & Poll Status
# ═══════════════════════════════════════════════════════════════


# ── POST /api/v1/analyze/video/start ───────────────────────────
@router.post("/start", response_model=VideoStartResponse)
async def start_video_analysis(
    body: VideoStartRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Dispatch the video ML inference task for a session.

    Preconditions:
    - Session must exist
    - Session status must be 'video_uploaded' (a video was uploaded in Sprint 3)

    The task runs asynchronously in a Celery worker. Use the
    /status/{session_uuid} endpoint to poll for results.
    """
    session_uuid = body.session_uuid

    # ── Validate session ────────────────────────────────────────
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

    # Must be in video_uploaded status to start analysis
    if session.status == "video_processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Video analysis is already in progress for this session.",
        )

    if session.status == "video_done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Video analysis has already been completed for this session.",
        )

    valid_start_statuses = {"video_uploaded", "error_video"}
    if session.status not in valid_start_statuses:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot start analysis when session status is '{session.status}'. "
            f"Video must be uploaded first (status must be one of: "
            f"{', '.join(sorted(valid_start_statuses))}).",
        )

    # ── Verify video file still exists ──────────────────────────
    session_dir = get_session_dir(session_uuid)
    video_exists = session_dir.exists() and any(
        f.name.startswith("upload") for f in session_dir.iterdir()
    ) if session_dir.exists() else False

    if not video_exists:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Video file not found. It may have expired. Please re-upload.",
        )

    # ── Dispatch Celery task ────────────────────────────────────
    from app.tasks.video_task import process_video

    task = process_video.delay(str(session_uuid))

    # ── Update session in DB ────────────────────────────────────
    session.celery_task_id = task.id
    session.status = "video_processing"
    session.video_error = None
    await db.commit()

    logger.info(
        "Video analysis dispatched: session=%s task_id=%s",
        anonymize_uuid(session_uuid),
        task.id[:8] + "***",
    )

    return VideoStartResponse(
        session_uuid=session_uuid,
        task_id=task.id,
        status="video_processing",
    )


# ── GET /api/v1/analyze/video/status/{session_uuid} ───────────
@router.get("/status/{session_uuid}", response_model=VideoStatusResponse)
async def get_video_status(
    session_uuid: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Poll the status of video ML inference for a session.

    Returns current status from the DB. If a Celery task ID is stored,
    also checks the Celery task state for terminal states that may not
    have been written to the DB yet.
    """
    # ── Load session ────────────────────────────────────────────
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

    # ── Check Celery task state if task is in-progress ──────────
    task_state = None
    if session.celery_task_id and session.status == "video_processing":
        try:
            from celery.result import AsyncResult
            from celery_worker import celery_app

            result_obj = AsyncResult(session.celery_task_id, app=celery_app)
            task_state = result_obj.state

            # If Celery reports FAILURE but DB still says processing,
            # update the DB to reflect the error
            if task_state == "FAILURE":
                session.status = "error_video"
                session.video_error = "Task failed in Celery worker"
                await db.commit()
        except Exception as e:
            logger.warning("Could not check Celery state: %s", e)

    return VideoStatusResponse(
        session_uuid=session_uuid,
        status=session.status,
        task_id=session.celery_task_id,
        task_state=task_state,
        video_class_probabilities=session.video_class_probabilities,
        video_confidence=session.video_confidence,
        video_score=session.video_score,
        risk_level=session.risk_level,
        error=session.video_error,
    )
