"""Test guest session creation directly."""
import asyncio
from app.db.session import AsyncSessionLocal
from app.models.session import AssessmentSession


async def test_guest():
    """Test guest session creation."""
    async with AsyncSessionLocal() as db:
        session = AssessmentSession(status="pending")
        db.add(session)
        await db.commit()
        await db.refresh(session)
        print(f"Session created: {session.session_uuid}")
        print(f"Status: {session.status}")


if __name__ == "__main__":
    asyncio.run(test_guest())
