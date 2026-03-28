"""
celery_worker.py
================
Celery application factory for AutiSense async tasks.

Usage:
    celery -A celery_worker.celery_app worker --loglevel=info
    celery -A celery_worker.celery_app beat --loglevel=info
"""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "autisense",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.video_task",         # Sprint 4 — video ML inference
        # "app.tasks.retention_task",   # Sprint 9 — data retention
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Safety limits — driven by config.py, not hardcoded
    task_soft_time_limit=settings.VIDEO_INFERENCE_SOFT_TIMEOUT,
    task_time_limit=settings.VIDEO_INFERENCE_HARD_TIMEOUT,
    worker_max_tasks_per_child=50,     # restart worker every 50 tasks (memory safety)
)

# Celery Beat schedule will be populated in Sprint 9:
# celery_app.conf.beat_schedule = { ... }
