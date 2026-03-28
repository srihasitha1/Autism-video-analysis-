"""
tests/conftest.py
=================
Shared pytest fixtures for the AutiSense test suite.

Uses an in-memory SQLite database for tests (no PostgreSQL required).
Each test function gets a fresh database so tests don't interfere.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.base import Base
from app.db.session import get_db

# We need to import all models so Base.metadata includes their tables.
from app.models import AssessmentSession, User  # noqa: F401


@pytest.fixture
async def test_app():
    """
    Async HTTP client with an isolated in-memory SQLite database.
    Creates all tables before the test and drops them after.
    """
    # ── In-memory SQLite (async via aiosqlite) ──────────────────
    engine = create_async_engine(
        "sqlite+aiosqlite://",  # in-memory
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSession = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    # Override the get_db dependency to use our test DB
    async def _override_get_db():
        async with TestSession() as session:
            try:
                yield session
            finally:
                await session.close()

    from app.main import app

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    # Cleanup
    app.dependency_overrides.clear()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def sample_session_uuid() -> str:
    """Generate a fresh UUID4 string for test sessions."""
    return str(uuid.uuid4())
