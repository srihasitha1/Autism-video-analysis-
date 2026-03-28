"""
app/routers/auth.py
===================
Authentication & session management endpoints.

Endpoints:
  POST /api/v1/auth/register  — Create account (email hashed, never stored plaintext)
  POST /api/v1/auth/login     — Login, returns JWT
  POST /api/v1/auth/guest     — Create anonymous assessment session
  GET  /api/v1/session/{uuid} — Get session status
  DELETE /api/v1/session/{uuid} — Delete session (GDPR right to erasure)
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.models.session import AssessmentSession
from app.models.user import User
from app.schemas.auth import (
    GuestSessionResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from app.schemas.session import SessionDeleteResponse, SessionStatusResponse
from app.services.auth_service import (
    create_access_token,
    hash_email,
    hash_password,
    verify_password,
)

logger = logging.getLogger("autisense.auth")

router = APIRouter(prefix="/auth", tags=["Authentication"])
session_router = APIRouter(tags=["Session Management"])


# ── POST /api/v1/auth/register ──────────────────────────────────
@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Create a new user account.
    Email is SHA-256 hashed before storage — never stored in plaintext.
    """
    email_hashed = hash_email(body.email)

    # Check uniqueness
    existing = await db.execute(
        select(User).where(User.email_hash == email_hashed)
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    # Create user
    user = User(
        email_hash=email_hashed,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Generate JWT
    token = create_access_token(data={"sub": str(user.user_uuid)})
    logger.info("User registered: %s***", str(user.user_uuid)[:8])

    return TokenResponse(
        access_token=token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── POST /api/v1/auth/login ────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate and receive a JWT.
    All auth errors return identical 'Invalid credentials' to prevent enumeration.
    """
    email_hashed = hash_email(body.email)

    result = await db.execute(
        select(User).where(User.email_hash == email_hashed)
    )
    user = result.scalar_one_or_none()

    # Identical error for wrong email OR wrong password
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # Update last_login
    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    token = create_access_token(data={"sub": str(user.user_uuid)})
    logger.info("User logged in: %s***", str(user.user_uuid)[:8])

    return TokenResponse(
        access_token=token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── POST /api/v1/auth/guest ────────────────────────────────────
@router.post("/guest", response_model=GuestSessionResponse)
async def create_guest_session(db: AsyncSession = Depends(get_db)):
    """
    Create an anonymous assessment session. No authentication required.
    Returns a session_uuid to track the assessment lifecycle.
    """
    session = AssessmentSession(status="pending")
    db.add(session)
    await db.commit()
    await db.refresh(session)

    logger.info("Guest session created: %s***", str(session.session_uuid)[:8])

    return GuestSessionResponse(
        session_uuid=session.session_uuid,
        created_at=session.created_at,
        status=session.status,
    )


# ── GET /api/v1/session/{session_uuid} ─────────────────────────
@session_router.get(
    "/session/{session_uuid}",
    response_model=SessionStatusResponse,
)
async def get_session_status(
    session_uuid: UUID, db: AsyncSession = Depends(get_db)
):
    """Get the current status of an assessment session."""
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

    return SessionStatusResponse(
        session_uuid=session.session_uuid,
        status=session.status,
        risk_level=session.risk_level,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


# ── DELETE /api/v1/session/{session_uuid} ──────────────────────
@session_router.delete(
    "/session/{session_uuid}",
    response_model=SessionDeleteResponse,
)
async def delete_session(
    session_uuid: UUID, db: AsyncSession = Depends(get_db)
):
    """
    Permanently delete a session and its temp files.
    Implements GDPR 'right to erasure'.
    """
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

    # Clean up any temp video files
    temp_dir = Path(settings.TEMP_VIDEO_DIR) / str(session_uuid)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Temp files cleaned for session: %s***", str(session_uuid)[:8])

    await db.delete(session)
    await db.commit()

    logger.info("Session deleted (GDPR): %s***", str(session_uuid)[:8])

    return SessionDeleteResponse(session_uuid=session_uuid)
