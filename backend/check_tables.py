"""Check database tables."""
import asyncio
from sqlalchemy import text
from app.db.session import engine


async def check_tables():
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        tables = [r[0] for r in result]
        print("Tables in database:", tables)
        if not tables:
            print("No tables found! Need to run migrations.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(check_tables())
