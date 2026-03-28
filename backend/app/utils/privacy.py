"""
app/utils/privacy.py
====================
Privacy-focused file handling utilities.

Principles:
  - Stream uploads in chunks (never load full video into RAM)
  - Isolate each session's files in its own directory
  - Enforce actual file size during streaming (not just Content-Length)
  - Secure deletion: overwrite bytes before unlinking
  - Anonymize UUIDs in logs (show only first 8 chars)
  - Clean up stale temp directories on a schedule
"""

import logging
import os
import shutil
import time
from pathlib import Path
from uuid import UUID

from app.config import settings

logger = logging.getLogger("autisense.privacy")

# ── Chunk size for streaming writes (1 MB) ──────────────────────
CHUNK_SIZE = 1024 * 1024  # 1 MB


def anonymize_uuid(uuid_val: UUID | str) -> str:
    """Show only first 8 characters of a UUID in logs."""
    return f"{str(uuid_val)[:8]}***"


def get_session_dir(session_uuid: UUID | str) -> Path:
    """Get the per-session temp directory path."""
    # Prevent path traversal: force the UUID to a safe basename
    safe_name = str(session_uuid).replace("/", "").replace("\\", "").replace("..", "")
    return Path(settings.TEMP_VIDEO_DIR) / safe_name


async def save_upload_to_temp(
    file_obj,
    session_uuid: UUID | str,
    extension: str,
    max_size_mb: int,
) -> Path:
    """
    Stream an uploaded file to a per-session temp directory.

    Writes in CHUNK_SIZE chunks to avoid loading the entire video into RAM.
    Enforces the actual byte count during streaming (not just Content-Length).

    Args:
        file_obj: The UploadFile.file (SpooledTemporaryFile).
        session_uuid: UUID of the assessment session.
        extension: Validated file extension (e.g., ".mp4").
        max_size_mb: Maximum allowed file size in MB.

    Returns:
        Path to the saved video file.

    Raises:
        ValueError: If actual file size exceeds max during streaming.
    """
    max_bytes = max_size_mb * 1024 * 1024

    # Create per-session directory
    session_dir = get_session_dir(session_uuid)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Output file path — use a safe fixed name
    video_path = session_dir / f"upload{extension}"

    total_written = 0

    try:
        with open(video_path, "wb") as out_file:
            while True:
                chunk = await file_obj.read(CHUNK_SIZE)
                if not chunk:
                    break

                total_written += len(chunk)

                # Enforce actual size limit during streaming
                if total_written > max_bytes:
                    raise ValueError(
                        f"File exceeds {max_size_mb} MB limit during upload. "
                        f"Received {total_written / (1024*1024):.1f} MB so far."
                    )

                out_file.write(chunk)
    except ValueError:
        # Clean up partial file on size violation
        secure_delete_file(video_path)
        raise
    except Exception:
        # Clean up on any unexpected error
        secure_delete_file(video_path)
        raise

    if total_written == 0:
        secure_delete_file(video_path)
        raise ValueError("Uploaded file is empty (0 bytes)")

    logger.info(
        "Video saved: session=%s size=%.1f MB path=%s",
        anonymize_uuid(session_uuid),
        total_written / (1024 * 1024),
        video_path.name,  # Only log filename, never full path
    )

    return video_path


def secure_delete_file(filepath: Path) -> None:
    """
    Securely delete a file by overwriting its contents with zeros before unlinking.
    Fails silently if file doesn't exist (safe for cleanup-on-error patterns).
    """
    try:
        if filepath.exists() and filepath.is_file():
            size = filepath.stat().st_size
            with open(filepath, "wb") as f:
                f.write(b"\x00" * size)
                f.flush()
                os.fsync(f.fileno())
            filepath.unlink()
            logger.debug("Securely deleted: %s", filepath.name)
    except Exception as e:
        logger.warning("Failed to securely delete %s: %s", filepath.name, e)


def cleanup_session_video(session_uuid: UUID | str) -> bool:
    """
    Remove the entire temp directory for a session.

    Returns:
        True if the directory existed and was cleaned up.
    """
    session_dir = get_session_dir(session_uuid)

    if session_dir.exists():
        # Securely delete individual files first, then remove the dir
        for file_path in session_dir.iterdir():
            if file_path.is_file():
                secure_delete_file(file_path)
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.info(
            "Session temp cleaned: %s", anonymize_uuid(session_uuid)
        )
        return True

    return False


def cleanup_stale_videos() -> int:
    """
    Scan the temp video directory and delete session dirs older than
    VIDEO_AUTO_DELETE_MINUTES.

    Called by APScheduler on a regular interval.

    Returns:
        Number of stale directories cleaned.
    """
    temp_root = Path(settings.TEMP_VIDEO_DIR)
    if not temp_root.exists():
        return 0

    cutoff = time.time() - (settings.VIDEO_AUTO_DELETE_MINUTES * 60)
    cleaned = 0

    for session_dir in temp_root.iterdir():
        if not session_dir.is_dir():
            continue

        try:
            dir_mtime = session_dir.stat().st_mtime
            if dir_mtime < cutoff:
                # Securely delete files first
                for file_path in session_dir.iterdir():
                    if file_path.is_file():
                        secure_delete_file(file_path)
                shutil.rmtree(session_dir, ignore_errors=True)
                cleaned += 1
                logger.info(
                    "Stale session cleaned: %s (age: %d min)",
                    session_dir.name[:8] + "***",
                    (time.time() - dir_mtime) / 60,
                )
        except Exception as e:
            logger.warning("Error cleaning stale dir %s: %s", session_dir.name[:8], e)

    if cleaned > 0:
        logger.info("Stale video cleanup: removed %d directories", cleaned)

    return cleaned
