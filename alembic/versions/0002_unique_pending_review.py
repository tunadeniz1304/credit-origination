"""one pending review per application

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-25 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

PENDING = sa.text("status = 'ONAY_BEKLIYOR'")


def upgrade() -> None:
    # Older duplicates (from the race) are closed before the index is created.
    op.execute(
        """
        UPDATE reviews SET status = 'IPTAL'
        WHERE status = 'ONAY_BEKLIYOR' AND id NOT IN (
            SELECT MIN(id) FROM reviews WHERE status = 'ONAY_BEKLIYOR' GROUP BY application_id
        )
        """
    )
    op.create_index(
        "uq_reviews_one_pending",
        "reviews",
        ["application_id"],
        unique=True,
        sqlite_where=PENDING,
        postgresql_where=PENDING,
    )


def downgrade() -> None:
    op.drop_index("uq_reviews_one_pending", table_name="reviews")
