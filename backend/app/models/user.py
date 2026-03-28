"""
app/models/user.py
==================
ORM model for registered users.

HARD RULE: No plaintext email, no name, no DOB, no phone, no address — ever.
Email is stored ONLY as a SHA-256 hash for lookup.
"""

import uuid

from sqlalchemy import Boolean, Column, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class User(Base):
    """
    Registered user account. Designed for privacy:
    - email_hash: SHA-256 of the lowercased email (never plaintext)
    - hashed_password: bcrypt hash
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_uuid = Column(
        UUID(as_uuid=True),
        unique=True,
        nullable=False,
        default=uuid.uuid4,
    )
    email_hash = Column(String(64), unique=True, nullable=False)  # SHA-256 hex digest
    hashed_password = Column(String(128), nullable=False)  # bcrypt
    display_name = Column(String(100), nullable=True)  # User-chosen display name

    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_login = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        uid = str(self.user_uuid)[:8] if self.user_uuid else "?"
        return f"<User {uid}*** active={self.is_active}>"
