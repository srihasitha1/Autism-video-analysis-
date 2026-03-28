"""
app/schemas/session.py
======================
Pydantic schemas for session status and management.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class SessionStatusResponse(BaseModel):
    """Current status of an assessment session."""
    session_uuid: UUID
    status: str
    risk_level: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SessionDeleteResponse(BaseModel):
    """Confirmation of session deletion (GDPR right to erasure)."""
    session_uuid: UUID
    message: str = "Session data permanently deleted"
