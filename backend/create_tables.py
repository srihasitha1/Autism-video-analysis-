"""Create tables directly using SQLAlchemy models."""
import asyncio
from sqlalchemy import text
from app.db.session import engine
from app.db.base import Base
from app.models.user import User  # noqa: F401
from app.models.session import AssessmentSession  # noqa: F401


async def create_tables():
    """Create all tables."""
    print("Creating tables...")
    async with engine.begin() as conn:
        # First drop alembic_version if exists to start fresh
        await conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
        # Create all tables from models
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully!")
    
    # Verify
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        tables = [r[0] for r in result]
        print("Tables in database:", tables)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(create_tables())
