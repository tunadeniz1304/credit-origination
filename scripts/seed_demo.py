"""Seed the six demo scenarios into the configured database.

Usage: python scripts/seed_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.users import ensure_demo_users
from app.db.session import init_db, session_scope
from app.demo.seed import seed_demo


def main() -> int:
    init_db()
    with session_scope() as session:
        ensure_demo_users(session)
    results = seed_demo()
    for r in results:
        print(
            f"{'OK ' if r['as_expected'] else 'XX '} {r['scenario']:12s} {r['application_id']} -> {r['state']}  ({r['title']})"
        )
    return 0 if all(r["as_expected"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
