"""
app/utils/validators.py
=======================
Video file validation utilities.

All uploads are treated as UNTRUSTED input. We validate:
  1. File extension — whitelist only (.mp4, .mov, .avi, .webm)
  2. File size — must not exceed MAX_VIDEO_SIZE_MB
  3. Magic bytes — verify first bytes match known video container signatures
     to prevent disguised files (e.g., .exe renamed to .mp4)
"""

import logging
from typing import BinaryIO

logger = logging.getLogger("autisense.validators")

# ── Allowed extensions ──────────────────────────────────────────
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}

# ── Magic bytes signatures ──────────────────────────────────────
# Each entry is (offset, bytes_to_match).
# We check if any signature matches at the given offset.
MAGIC_SIGNATURES = [
    # MP4 / MOV (ISO BMFF) — "ftyp" at offset 4
    (4, b"ftyp"),
    # AVI — "RIFF" header at offset 0
    (0, b"RIFF"),
    # WebM / MKV — EBML header at offset 0
    (0, b"\x1a\x45\xdf\xa3"),
    # MP4 free atom (some encoders put "free" before "ftyp")
    (4, b"free"),
    # MP4 mdat atom (raw data first, ftyp later)
    (4, b"mdat"),
    # MOV — "moov" atom at offset 4
    (4, b"moov"),
    # MOV — "wide" atom at offset 4
    (4, b"wide"),
]


def validate_extension(filename: str) -> str:
    """
    Check if the filename has an allowed video extension.

    Returns:
        The lowercased extension (e.g., ".mp4").

    Raises:
        ValueError: If extension is missing or not allowed.
    """
    if not filename or "." not in filename:
        raise ValueError("File must have an extension")

    ext = "." + filename.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File type '{ext}' is not allowed. "
            f"Accepted types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def validate_file_size(content_length: int | None, max_mb: int) -> None:
    """
    Validate the declared Content-Length against the maximum allowed size.

    Args:
        content_length: The Content-Length header value (bytes), may be None.
        max_mb: Maximum allowed size in megabytes.

    Raises:
        ValueError: If the file exceeds the maximum size.
    """
    max_bytes = max_mb * 1024 * 1024
    if content_length is not None and content_length > max_bytes:
        raise ValueError(
            f"File too large ({content_length / (1024*1024):.1f} MB). "
            f"Maximum allowed: {max_mb} MB"
        )


def validate_magic_bytes(header: bytes) -> bool:
    """
    Check if the file header matches any known video container signature.

    Args:
        header: The first 12+ bytes of the file.

    Returns:
        True if a known signature is found.

    Raises:
        ValueError: If no known video signature is detected.
    """
    if len(header) < 12:
        raise ValueError("File is too small to be a valid video")

    for offset, signature in MAGIC_SIGNATURES:
        end = offset + len(signature)
        if end <= len(header) and header[offset:end] == signature:
            return True

    raise ValueError(
        "File does not appear to be a valid video. "
        "The file header does not match any known video format."
    )


async def validate_upload_stream(
    file_obj: BinaryIO,
    filename: str,
    content_length: int | None,
    max_size_mb: int,
) -> str:
    """
    Run all validations on an uploaded file in a single pass.

    Args:
        file_obj: The file-like object (SpooledTemporaryFile from FastAPI).
        filename: The original filename.
        content_length: Declared Content-Length in bytes.
        max_size_mb: Maximum allowed size in MB.

    Returns:
        The validated, lowercased file extension.

    Raises:
        ValueError: On any validation failure.
    """
    # 1. Extension check
    ext = validate_extension(filename)

    # 2. Declared size check (fast, before any reads)
    validate_file_size(content_length, max_size_mb)

    # 3. Read first 12 bytes for magic byte check
    header = await file_obj.read(12)
    validate_magic_bytes(header)

    # Reset file position for subsequent reads
    await file_obj.seek(0)

    return ext
