"""add video inference columns to assessment_sessions

Revision ID: 002_add_video_inference_cols
Revises: 001_initial_tables
Create Date: 2026-03-28

Adds celery_task_id, video_score, and video_error columns
to the assessment_sessions table for Sprint 4 video inference.

NOTE: Made idempotent to handle cases where columns already exist
(e.g., if 001_initial_tables already includes these columns).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# Revision identifiers
revision = "002_add_video_inference_cols"
down_revision = '001_initial_tables'
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column already exists in the table."""
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    # Add video_score if not exists
    if not _column_exists("assessment_sessions", "video_score"):
        op.add_column(
            "assessment_sessions",
            sa.Column("video_score", sa.Float(), nullable=True),
        )
    
    # Add video_error if not exists
    if not _column_exists("assessment_sessions", "video_error"):
        op.add_column(
            "assessment_sessions",
            sa.Column("video_error", sa.String(500), nullable=True),
        )
    
    # Add celery_task_id if not exists
    if not _column_exists("assessment_sessions", "celery_task_id"):
        op.add_column(
            "assessment_sessions",
            sa.Column("celery_task_id", sa.String(255), nullable=True),
        )


def downgrade() -> None:
    if _column_exists("assessment_sessions", "celery_task_id"):
        op.drop_column("assessment_sessions", "celery_task_id")
    if _column_exists("assessment_sessions", "video_error"):
        op.drop_column("assessment_sessions", "video_error")
    if _column_exists("assessment_sessions", "video_score"):
        op.drop_column("assessment_sessions", "video_score")
