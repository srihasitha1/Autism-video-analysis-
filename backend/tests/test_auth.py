"""
tests/test_auth.py
==================
Sprint 2 verification: auth endpoints, session management, and security.
"""

import uuid


# ────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────
def _unique_email() -> str:
    """Generate a unique test email to avoid collisions between tests."""
    return f"test-{uuid.uuid4().hex[:8]}@example.com"


# ────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────
async def test_register_success(test_app):
    """POST /api/v1/auth/register with valid data returns 201 + JWT."""
    resp = await test_app.post(
        "/api/v1/auth/register",
        json={"email": _unique_email(), "password": "StrongPass1"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0


async def test_register_weak_password_no_uppercase(test_app):
    """Password missing uppercase should fail validation."""
    resp = await test_app.post(
        "/api/v1/auth/register",
        json={"email": _unique_email(), "password": "weakpass1"},
    )
    assert resp.status_code == 422


async def test_register_weak_password_no_number(test_app):
    """Password missing number should fail validation."""
    resp = await test_app.post(
        "/api/v1/auth/register",
        json={"email": _unique_email(), "password": "WeakPasss"},
    )
    assert resp.status_code == 422


async def test_register_weak_password_too_short(test_app):
    """Password < 8 chars should fail validation."""
    resp = await test_app.post(
        "/api/v1/auth/register",
        json={"email": _unique_email(), "password": "Sh0rt"},
    )
    assert resp.status_code == 422


async def test_register_duplicate_email(test_app):
    """Registering the same email twice should return 409."""
    email = _unique_email()
    # First registration
    resp1 = await test_app.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "StrongPass1"},
    )
    assert resp1.status_code == 201

    # Duplicate
    resp2 = await test_app.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "StrongPass1"},
    )
    assert resp2.status_code == 409


# ────────────────────────────────────────────────────────────────
# Login
# ────────────────────────────────────────────────────────────────
async def test_login_success(test_app):
    """POST /api/v1/auth/login with valid credentials returns JWT."""
    email = _unique_email()
    # Register first
    await test_app.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "StrongPass1"},
    )

    # Login
    resp = await test_app.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "StrongPass1"},
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()


async def test_login_wrong_password(test_app):
    """Wrong password returns 401 with generic error."""
    email = _unique_email()
    await test_app.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "StrongPass1"},
    )

    resp = await test_app.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "WrongPass1"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid credentials"


async def test_login_nonexistent_email(test_app):
    """Login with unknown email returns 401 (same error, no enumeration)."""
    resp = await test_app.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "StrongPass1"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid credentials"


# ────────────────────────────────────────────────────────────────
# Guest Session
# ────────────────────────────────────────────────────────────────
async def test_create_guest_session(test_app):
    """POST /api/v1/auth/guest creates a session with pending status."""
    resp = await test_app.post("/api/v1/auth/guest")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_uuid" in data
    assert data["status"] == "pending"


# ────────────────────────────────────────────────────────────────
# Session Status & Deletion
# ────────────────────────────────────────────────────────────────
async def test_get_session_status(test_app):
    """GET /api/v1/session/{uuid} returns session info."""
    # Create guest session
    create_resp = await test_app.post("/api/v1/auth/guest")
    session_uuid = create_resp.json()["session_uuid"]

    # Get status
    resp = await test_app.get(f"/api/v1/session/{session_uuid}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"


async def test_get_session_not_found(test_app):
    """GET /api/v1/session/{bad_uuid} returns 404."""
    resp = await test_app.get(f"/api/v1/session/{uuid.uuid4()}")
    assert resp.status_code == 404


async def test_delete_session(test_app):
    """DELETE /api/v1/session/{uuid} removes session (GDPR)."""
    # Create
    create_resp = await test_app.post("/api/v1/auth/guest")
    session_uuid = create_resp.json()["session_uuid"]

    # Delete
    del_resp = await test_app.delete(f"/api/v1/session/{session_uuid}")
    assert del_resp.status_code == 200
    assert "deleted" in del_resp.json()["message"].lower()

    # Verify gone
    get_resp = await test_app.get(f"/api/v1/session/{session_uuid}")
    assert get_resp.status_code == 404


async def test_delete_session_not_found(test_app):
    """DELETE /api/v1/session/{bad_uuid} returns 404."""
    resp = await test_app.delete(f"/api/v1/session/{uuid.uuid4()}")
    assert resp.status_code == 404


# ────────────────────────────────────────────────────────────────
# Security
# ────────────────────────────────────────────────────────────────
async def test_auth_errors_are_identical(test_app):
    """Both wrong email and wrong password return the same error message."""
    email = _unique_email()
    await test_app.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "StrongPass1"},
    )

    wrong_email = await test_app.post(
        "/api/v1/auth/login",
        json={"email": "wrong@example.com", "password": "StrongPass1"},
    )
    wrong_pass = await test_app.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "WrongPass1"},
    )

    # Both should return 401 with identical detail
    assert wrong_email.status_code == wrong_pass.status_code == 401
    assert wrong_email.json()["detail"] == wrong_pass.json()["detail"] == "Invalid credentials"
