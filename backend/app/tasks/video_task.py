"""
app/tasks/video_task.py
=======================
Celery task for asynchronous video ML inference.

Flow:
  1. Validate session exists and is in correct status (idempotency guard)
  2. Update status to video_processing
  3. Locate uploaded video file in session temp directory
  4. Call run_inference() from the adapter service
  5. Store results in DB (video_class_probabilities, video_score, etc.)
  6. Update status to video_done
  7. ALWAYS clean up temp video files in finally block

Error handling:
  - On any exception, set status to error_video and store error message
  - Celery soft_time_limit triggers SoftTimeLimitExceeded
  - Celery time_limit hard-kills the task
  - Retries up to 2 times on transient errors, NOT on validation errors
"""

import logging
import traceback
from pathlib import Path
from uuid import UUID

from celery import states
from celery.exceptions import SoftTimeLimitExceeded

from celery_worker import celery_app
from app.config import settings

logger = logging.getLogger("autisense.video_task")


def _find_video_file(session_uuid: str) -> Path | None:
    """Locate the uploaded video file in the session temp directory."""
    from app.utils.privacy import get_session_dir

    session_dir = get_session_dir(session_uuid)
    if not session_dir.exists():
        return None

    # Sprint 3 saves as upload.<ext>
    for f in session_dir.iterdir():
        if f.is_file() and f.name.startswith("upload"):
            return f

    return None


@celery_app.task(
    bind=True,
    name="app.tasks.video_task.process_video",
    max_retries=2,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=settings.VIDEO_INFERENCE_SOFT_TIMEOUT,
    time_limit=settings.VIDEO_INFERENCE_HARD_TIMEOUT,
)
def process_video(self, session_uuid: str) -> dict:
    """
    Async Celery task: run video ML inference for a session.

    Args:
        session_uuid: UUID string of the assessment session.

    Returns:
        dict with task result (status, scores, or error).
    """
    from app.db.sync_session import SyncSessionLocal
    from app.models.session import AssessmentSession
    from app.utils.privacy import cleanup_session_video, anonymize_uuid
    from sqlalchemy import select

    anon_uuid = anonymize_uuid(session_uuid)
    logger.info("Starting video task for session %s", anon_uuid)

    db = SyncSessionLocal()
    try:
        # ── 0. Convert UUID string to UUID object ───────────────
        try:
            session_uuid_obj = UUID(session_uuid) if isinstance(session_uuid, str) else session_uuid
        except ValueError:
            logger.error("Invalid UUID format: %s", session_uuid)
            return {"status": "error", "error": "Invalid UUID format"}

        # ── 1. Load session from DB ─────────────────────────────
        result = db.execute(
            select(AssessmentSession).where(
                AssessmentSession.session_uuid == session_uuid_obj
            )
        )
        session = result.scalar_one_or_none()

        if session is None:
            logger.error("Session not found: %s", anon_uuid)
            return {"status": "error", "error": "Session not found"}

        # ── 2. Idempotency guard ────────────────────────────────
        if session.status == "video_done":
            logger.info("Session %s already processed, skipping", anon_uuid)
            return {
                "status": "already_done",
                "video_score": session.video_score,
                "video_confidence": session.video_confidence,
            }

        if session.status not in ("video_uploaded", "video_processing", "error_video"):
            logger.warning(
                "Session %s in unexpected status '%s', cannot process",
                anon_uuid, session.status,
            )
            return {
                "status": "error",
                "error": f"Invalid session status: {session.status}",
            }

        # ── 3. Mark as processing ───────────────────────────────
        session.status = "video_processing"
        session.video_error = None  # Clear previous error
        db.commit()

        # ── 4. Find the uploaded video file ─────────────────────
        video_path = _find_video_file(session_uuid)
        if video_path is None:
            raise FileNotFoundError(
                "Uploaded video file not found in temp directory. "
                "It may have been cleaned up before processing started."
            )

        logger.info(
            "Found video file for session %s: %s",
            anon_uuid, video_path.name,
        )

        # ── 5. Run ML inference ─────────────────────────────────
        from app.services.video_inference import run_inference

        inference_result = run_inference(str(video_path))

        # ── 6. Store results in DB ──────────────────────────────
        session.video_class_probabilities = inference_result["video_class_probabilities"]
        session.video_score = inference_result["video_score"]
        session.video_confidence = inference_result["video_confidence"]
        session.risk_level = inference_result["risk_level"]
        session.status = "video_done"
        session.video_error = None
        db.commit()

        logger.info(
            "Video inference complete for session %s: score=%.4f risk=%s",
            anon_uuid,
            inference_result["video_score"],
            inference_result["risk_level"],
        )

        return {
            "status": "video_done",
            "video_score": inference_result["video_score"],
            "video_confidence": inference_result["video_confidence"],
            "risk_level": inference_result["risk_level"],
            "clips_evaluated": inference_result["clips_evaluated"],
        }

    except SoftTimeLimitExceeded:
        logger.error("Video task TIMED OUT for session %s", anon_uuid)
        _mark_error(db, session_uuid, "Video analysis timed out. Please try with a shorter video.")
        return {"status": "error_video", "error": "Timed out"}

    except FileNotFoundError as e:
        logger.error("File not found for session %s: %s", anon_uuid, str(e))
        _mark_error(db, session_uuid, str(e))
        return {"status": "error_video", "error": str(e)}

    except Exception as e:
        logger.error(
            "Video task failed for session %s: %s\n%s",
            anon_uuid, str(e), traceback.format_exc(),
        )
        error_msg = f"Video analysis failed: {type(e).__name__}"
        _mark_error(db, session_uuid, error_msg)

        # Retry on transient errors (not validation/file errors)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=10 * (self.request.retries + 1))

        return {"status": "error_video", "error": error_msg}

    finally:
        # ── ALWAYS clean up temp files ──────────────────────────
        try:
            cleaned = cleanup_session_video(session_uuid)
            if cleaned:
                logger.info("Temp files cleaned for session %s", anon_uuid)
            else:
                logger.debug("No temp files to clean for session %s", anon_uuid)
        except Exception as cleanup_err:
            logger.warning(
                "Cleanup failed for session %s: %s",
                anon_uuid, cleanup_err,
            )

        db.close()


def _mark_error(db, session_uuid: str, error_msg: str):
    """Helper to mark a session as errored in the DB."""
    from app.models.session import AssessmentSession
    from sqlalchemy import select

    try:
        session_uuid_obj = UUID(session_uuid) if isinstance(session_uuid, str) else session_uuid
        result = db.execute(
            select(AssessmentSession).where(
                AssessmentSession.session_uuid == session_uuid_obj
            )
        )
        session = result.scalar_one_or_none()
        if session:
            session.status = "error_video"
            session.video_error = error_msg[:500]  # Truncate to column limit
            db.commit()
    except Exception as e:
        logger.error("Failed to mark session error: %s", e)
        db.rollback()
