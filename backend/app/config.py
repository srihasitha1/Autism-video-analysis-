"""
app/config.py
=============
Centralized configuration via pydantic-settings.
All values loaded from .env file or environment variables.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — all loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://autisense:password@localhost:5432/autisense"

    # ── Auth / JWT ──────────────────────────────────────────────
    SECRET_KEY: str = "change-me-to-a-real-secret-key-at-least-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── External APIs ───────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GOOGLE_PLACES_API_KEY: str = ""

    # ── Redis / Celery ──────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── Video processing ────────────────────────────────────────
    TEMP_VIDEO_DIR: str = "/tmp/autisense_videos"
    VIDEO_AUTO_DELETE_MINUTES: int = 30
    MAX_VIDEO_SIZE_MB: int = 50
    MAX_VIDEO_DURATION_SECONDS: int = 10
    MIN_VIDEO_DURATION_SECONDS: int = 1

    # ── ML model paths ──────────────────────────────────────────
    MODEL_VIDEO_PATH: str = "../model/autism_final.h5"
    MODEL_ENCODER_PATH: str = "ml_models/video_model/label_encoder.pkl"
    MODEL_RF_PATH: str = "ml_models/questionnaire_model/autism_model.pkl"

    # ── Video inference timeouts ─────────────────────────────────
    VIDEO_INFERENCE_SOFT_TIMEOUT: int = 300   # seconds — Celery soft limit (5 min)
    VIDEO_INFERENCE_HARD_TIMEOUT: int = 600   # seconds — Celery hard kill (10 min)

    # ── CORS ────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── App metadata ────────────────────────────────────────────
    ENVIRONMENT: str = "development"  # development | staging | production
    LOG_LEVEL: str = "INFO"
    APP_VERSION: str = "1.0.0"


# Module-level singleton — import this everywhere.
settings = Settings()
