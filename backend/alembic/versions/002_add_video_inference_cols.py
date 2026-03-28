"""add video inference columns to assessment_sessions

Revision ID: 002_add_video_inference_cols
Revises: 001_initial_tables
Create Date: 2026-03-28

Adds celery_task_id, video_score, and video_error columns
to the assessment_sessions table for Sprint 4 video inference.
"""
from alembic import op
import sqlalchemy as sa


# Revision identifiers
revision = "002_add_video_inference_cols"
down_revision = '001_initial_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "assessment_sessions",
        sa.Column("video_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "assessment_sessions",
        sa.Column("video_error", sa.String(500), nullable=True),
    )
    op.add_column(
        "assessment_sessions",
        sa.Column("celery_task_id", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("assessment_sessions", "celery_task_id")
    op.drop_column("assessment_sessions", "video_error")
    op.drop_column("assessment_sessions", "video_score")
