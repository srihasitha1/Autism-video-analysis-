"""Direct API test without FastAPI router."""
import httpx
import asyncio


async def test_api():
    """Test the API endpoints."""
    async with httpx.AsyncClient() as client:
        # Test health
        resp = await client.get("http://localhost:8000/health")
        print(f"Health: {resp.status_code} - {resp.text}")
        
        # Test guest
        try:
            resp = await client.post("http://localhost:8000/api/v1/auth/guest")
            print(f"Guest: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Guest error: {e}")
        
        # Test register
        try:
            resp = await client.post(
                "http://localhost:8000/api/v1/auth/register",
                json={"email": "test123@test.com", "password": "TestPass123"},
            )
            print(f"Register: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Register error: {e}")


if __name__ == "__main__":
    asyncio.run(test_api())
