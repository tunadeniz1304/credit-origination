"""Enforce the per-area coverage floors of the quality gate on ``coverage.xml``.

Usage: ``python -m pytest --cov=app --cov-report=xml && python scripts/check_coverage.py``
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FLOORS = {
    "app/governance/": 0.80,
    "app/worker/": 0.80,
    "app/core/task_dispatcher.py": 0.80,
}


def main(path: str = "coverage.xml") -> int:
    root = ET.parse(Path(path)).getroot()
    totals = {prefix: [0, 0] for prefix in FLOORS}
    for cls in root.iter("class"):
        filename = "app/" + cls.get("filename", "").replace("\\", "/").removeprefix("app/")
        lines = cls.findall("lines/line")
        covered = sum(1 for line in lines if int(line.get("hits", "0")) > 0)
        for prefix, counts in totals.items():
            if filename.startswith(prefix):
                counts[0] += covered
                counts[1] += len(lines)
    failed = False
    for prefix, (covered, total) in totals.items():
        rate = covered / total if total else 0.0
        ok = total > 0 and rate >= FLOORS[prefix]
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {prefix:32} {rate:6.1%} (floor {FLOORS[prefix]:.0%})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
