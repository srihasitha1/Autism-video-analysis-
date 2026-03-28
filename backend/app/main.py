"""
app/main.py
===========
FastAPI application factory for AutiSense backend.

Includes:
- CORS middleware (origin-restricted)
- Security headers middleware
- Health check & ping endpoints
- Startup / shutdown lifecycle hooks
"""

import logging
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings

logger = logging.getLogger("autisense")


# ── Lifespan ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    # ── Startup ─────────────────────────────────────────────────
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create temp video directory
    temp_dir = Path(settings.TEMP_VIDEO_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Temp video directory ready: %s", temp_dir)

    logger.info(
        "AutiSense v%s starting [env=%s]",
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )

    yield  # ← application runs here

    # ── Shutdown ────────────────────────────────────────────────
    logger.info("AutiSense shutting down gracefully.")


# ── App factory ─────────────────────────────────────────────────
def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    application = FastAPI(
        title="AutiSense API",
        description=(
            "Privacy-first, AI-powered early behavioral risk assessment platform. "
            "This is NOT a diagnostic tool."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── Exception handlers ──────────────────────────────────────
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc)
        logger.error("Traceback: %s", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "type": type(exc).__name__},
        )

    # ── Security headers middleware ─────────────────────────────
    @application.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=()"
        # Remove server header to avoid leaking stack info
        if "server" in response.headers:
            del response.headers["server"]
        return response

    # ── Root-level endpoints ────────────────────────────────────
    @application.get("/health", tags=["System"])
    async def health_check():
        """Liveness probe — returns 200 if the service is up."""
        return {
            "status": "ok",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    @application.get("/api/v1/ping", tags=["System"])
    async def ping():
        """Lightweight connectivity check."""
        return {"pong": True}

    # ── Include routers ────────────────────────────────────────
    from app.routers.auth import router as auth_router, session_router
    from app.routers.video import router as video_router
    from app.routers.questionnaire import router as questionnaire_router
    from app.routers.fusion import router as fusion_router

    application.include_router(auth_router, prefix="/api/v1")
    application.include_router(session_router, prefix="/api/v1")
    application.include_router(video_router, prefix="/api/v1")
    application.include_router(questionnaire_router, prefix="/api/v1")
    application.include_router(fusion_router, prefix="/api/v1")

    # Future sprint routers:
    # from app.routers import chatbot, clinics
    # ...

    return application


# Module-level app instance — used by uvicorn & tests
app = create_app()
