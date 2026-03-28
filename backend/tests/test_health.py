"""
tests/test_health.py
====================
Sprint 1 verification: health check and ping endpoints.
"""


async def test_health_returns_200(test_app):
    """GET /health should return 200 with status, version, and environment."""
    response = await test_app.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data


async def test_ping_returns_pong(test_app):
    """GET /api/v1/ping should return 200 with pong=true."""
    response = await test_app.get("/api/v1/ping")
    assert response.status_code == 200

    data = response.json()
    assert data["pong"] is True


async def test_security_headers_present(test_app):
    """All responses must include security headers."""
    response = await test_app.get("/health")

    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "no-referrer"
    assert "camera=()" in response.headers.get("permissions-policy", "")


async def test_server_header_removed(test_app):
    """Server header should not be present in responses."""
    response = await test_app.get("/health")
    assert "server" not in response.headers


async def test_nonexistent_route_returns_404(test_app):
    """Unknown routes should return 404, not crash."""
    response = await test_app.get("/api/v1/does-not-exist")
    assert response.status_code == 404
