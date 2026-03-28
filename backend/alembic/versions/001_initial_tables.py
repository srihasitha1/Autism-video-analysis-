"""initial tables for users and assessment_sessions

Revision ID: 001_initial_tables
Revises: None
Create Date: 2026-03-28

Creates the base tables for the AutiSense application:
- users: registered user accounts (privacy-focused, email stored as hash only)
- assessment_sessions: anonymous assessment sessions with video + questionnaire data
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# Revision identifiers
revision = '001_initial_tables'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── CREATE TABLE users ──────────────────────────────────────────────────
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_uuid', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_hash', sa.String(length=64), nullable=False),
        sa.Column('hashed_password', sa.String(length=128), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_uuid'),
        sa.UniqueConstraint('email_hash'),
    )

    # ── CREATE TABLE assessment_sessions ────────────────────────────────────
    op.create_table(
        'assessment_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_uuid', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('child_age_months', sa.Integer(), nullable=True),
        sa.Column('child_gender', sa.String(length=20), nullable=True),
        sa.Column('video_class_probabilities', postgresql.JSON(), nullable=True),
        sa.Column('video_confidence', sa.String(length=20), nullable=True),
        sa.Column('video_score', sa.Float(), nullable=True),
        sa.Column('video_error', sa.String(length=500), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('questionnaire_raw_scores', postgresql.JSON(), nullable=True),
        sa.Column('questionnaire_probability', sa.Float(), nullable=True),
        sa.Column('category_scores', postgresql.JSON(), nullable=True),
        sa.Column('final_risk_score', sa.Float(), nullable=True),
        sa.Column('risk_level', sa.String(length=10), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('video_contribution', sa.String(length=20), nullable=True),
        sa.Column('questionnaire_contribution', sa.String(length=20), nullable=True),
        sa.Column('status', sa.String(length=25), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_uuid'),
    )

    # ── CREATE INDEXES ───────────────────────────────────────────────────────
    op.create_index('ix_assessment_sessions_session_uuid', 'assessment_sessions', ['session_uuid'])
    op.create_index('ix_assessment_sessions_status', 'assessment_sessions', ['status'])
    op.create_index('ix_assessment_sessions_created_at', 'assessment_sessions', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_assessment_sessions_created_at', 'assessment_sessions')
    op.drop_index('ix_assessment_sessions_status', 'assessment_sessions')
    op.drop_index('ix_assessment_sessions_session_uuid', 'assessment_sessions')
    op.drop_table('assessment_sessions')
    op.drop_table('users')
