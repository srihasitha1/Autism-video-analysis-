"""
app/schemas/auth.py
===================
Pydantic schemas for authentication & session management.
"""

import re
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, field_validator


class RegisterRequest(BaseModel):
    """New user registration."""
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least 1 uppercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least 1 number")
        return v


class LoginRequest(BaseModel):
    """User login."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class GuestSessionResponse(BaseModel):
    """Anonymous guest session creation response."""
    session_uuid: UUID
    created_at: datetime
    status: str


class UpdateProfileRequest(BaseModel):
    """Update user display name."""
    display_name: str

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 1:
            raise ValueError("Display name cannot be empty")
        if len(v) > 100:
            raise ValueError("Display name too long (max 100 characters)")
        return v


class UserProfileResponse(BaseModel):
    """Current user profile."""
    user_uuid: UUID
    display_name: Optional[str] = None
    is_guest: bool  # True if no auth token
    created_at: datetime
