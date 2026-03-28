"""add display_name to users and user_uuid to sessions

Revision ID: 004_add_user_profile_cols
Revises: 003_add_video_confidence_cols
Create Date: 2026-03-28

Adds:
- display_name column to users table
- user_uuid foreign key to assessment_sessions table
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "004_add_user_profile_cols"
down_revision = '003_add_video_confidence_cols'
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def _constraint_exists(table_name: str, constraint_name: str) -> bool:
    conn = op.get_bind()
    inspector = inspect(conn)
    foreign_keys = inspector.get_foreign_keys(table_name)
    return any(fk["name"] == constraint_name for fk in foreign_keys)


def upgrade() -> None:
    # Add display_name to users
    if not _column_exists("users", "display_name"):
        op.add_column(
            "users",
            sa.Column("display_name", sa.String(100), nullable=True),
        )
    
    # Add user_uuid to assessment_sessions
    if not _column_exists("assessment_sessions", "user_uuid"):
        op.add_column(
            "assessment_sessions",
            sa.Column(
                "user_uuid",
                sa.dialects.postgresql.UUID(as_uuid=True),
                nullable=True,
            ),
        )
        op.create_index(
            "ix_assessment_sessions_user_uuid",
            "assessment_sessions",
            ["user_uuid"],
        )
    
    # Add foreign key constraint
    if not _constraint_exists("assessment_sessions", "fk_sessions_user_uuid"):
        op.create_foreign_key(
            "fk_sessions_user_uuid",
            "assessment_sessions",
            "users",
            ["user_uuid"],
            ["user_uuid"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    if _constraint_exists("assessment_sessions", "fk_sessions_user_uuid"):
        op.drop_constraint(
            "fk_sessions_user_uuid",
            "assessment_sessions",
            type_="foreignkey",
        )
    
    if _column_exists("assessment_sessions", "user_uuid"):
        op.drop_index("ix_assessment_sessions_user_uuid", "assessment_sessions")
        op.drop_column("assessment_sessions", "user_uuid")
    
    if _column_exists("users", "display_name"):
        op.drop_column("users", "display_name")
