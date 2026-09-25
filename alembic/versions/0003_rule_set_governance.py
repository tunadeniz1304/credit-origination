"""rule-set submitter and one active rule set

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-25 18:00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

ACTIVE = sa.text("status = 'YURURLUKTE'")


def upgrade() -> None:
    with op.batch_alter_table("rule_sets") as batch:
        batch.add_column(sa.Column("submitted_by", sa.String(64), nullable=True))
    # Keep only the most recently activated rule set in force before indexing.
    op.execute(
        """
        UPDATE rule_sets SET status = 'ARSIV'
        WHERE status = 'YURURLUKTE' AND id NOT IN (
            SELECT id FROM rule_sets WHERE status = 'YURURLUKTE'
            ORDER BY activated_at DESC LIMIT 1
        )
        """
    )
    op.create_index(
        "uq_rule_sets_one_active",
        "rule_sets",
        ["status"],
        unique=True,
        sqlite_where=ACTIVE,
        postgresql_where=ACTIVE,
    )


def downgrade() -> None:
    op.drop_index("uq_rule_sets_one_active", table_name="rule_sets")
    with op.batch_alter_table("rule_sets") as batch:
        batch.drop_column("submitted_by")
