"""Test registration endpoint directly."""
import asyncio
from app.db.session import AsyncSessionLocal
from app.services.auth_service import hash_email, hash_password
from app.models.user import User
from sqlalchemy import select


async def test_register():
    """Test user registration."""
    email = "test@test.com"
    password = "Test123456"
    
    email_hashed = hash_email(email)
    print(f"Email hash: {email_hashed}")
    
    async with AsyncSessionLocal() as db:
        # Check if exists
        result = await db.execute(
            select(User).where(User.email_hash == email_hashed)
        )
        existing = result.scalar_one_or_none()
        if existing:
            print("User already exists!")
            return
        
        # Create user
        user = User(
            email_hash=email_hashed,
            hashed_password=hash_password(password),
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        print(f"User created: {user.user_uuid}")
        print(f"User is_active: {user.is_active}")


if __name__ == "__main__":
    asyncio.run(test_register())
