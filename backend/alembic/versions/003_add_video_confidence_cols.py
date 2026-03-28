"""add video confidence score and variance columns

Revision ID: 003_add_video_confidence_cols
Revises: 002_add_video_inference_cols
Create Date: 2026-03-28

Adds video_confidence_score and video_variance columns
to the assessment_sessions table for dynamic weighting in fusion.

These columns enable the fusion engine to use numeric confidence
scores for more granular weight assignment between video and
questionnaire risk assessments.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# Revision identifiers
revision = "003_add_video_confidence_cols"
down_revision = '002_add_video_inference_cols'
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column already exists in the table."""
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    # Add video_confidence_score if not exists
    if not _column_exists("assessment_sessions", "video_confidence_score"):
        op.add_column(
            "assessment_sessions",
            sa.Column("video_confidence_score", sa.Float(), nullable=True),
        )
    
    # Add video_variance if not exists
    if not _column_exists("assessment_sessions", "video_variance"):
        op.add_column(
            "assessment_sessions",
            sa.Column("video_variance", sa.Float(), nullable=True),
        )


def downgrade() -> None:
    if _column_exists("assessment_sessions", "video_variance"):
        op.drop_column("assessment_sessions", "video_variance")
    if _column_exists("assessment_sessions", "video_confidence_score"):
        op.drop_column("assessment_sessions", "video_confidence_score")
