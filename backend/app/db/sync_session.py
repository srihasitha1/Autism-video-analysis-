"""
app/db/sync_session.py
======================
Synchronous SQLAlchemy engine + session for use inside Celery tasks.

WHY: Celery workers run synchronous Python — they cannot use the async
     engine/session from session.py. This module provides a sync alternative
     using psycopg2 as the DB driver.

ONLY use this inside Celery tasks. FastAPI endpoints continue using the
async engine from app.db.session.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


def _build_sync_url(async_url: str) -> str:
    """Convert an asyncpg DATABASE_URL to a psycopg2 one."""
    return async_url.replace("postgresql+asyncpg", "postgresql+psycopg2")


sync_engine = create_engine(
    _build_sync_url(settings.DATABASE_URL),
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
)
