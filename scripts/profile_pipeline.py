"""Profile one clean application end to end (inline mode) with pyinstrument.

Usage: python scripts/profile_pipeline.py [--runs 5] [--html out.html]

Starts the app in-process (TestClient, fresh temp DB), submits ``--runs``
clean applications with their five documents and prints the end-to-end
latency per application plus the pyinstrument profile of the last one.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_TMP = tempfile.mkdtemp(prefix="anil2-profile-")
os.environ.update(
    {
        "ANIL2_NO_DOTENV": "1",
        "APP_ENV": "test",
        "LLM_MODE": "demo",
        "LLM_API_KEY": "",
        "TASK_QUEUE_BACKEND": "inline",
        "CIRCUIT_STATE_BACKEND": "memory",
        "RATE_LIMIT_ENABLED": "false",
        "COOKIE_SECURE": "false",
        "DATABASE_URL": f"sqlite:///{Path(_TMP, 'p.db').as_posix()}",
        "REPORT_OUTPUT_DIR": str(Path(_TMP, "reports")),
        "RESULT_STORE_DIR": str(Path(_TMP, "results")),
        "UPLOAD_DIR": str(Path(_TMP, "uploads")),
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--html", default="")
    args = parser.parse_args()
    from fastapi.testclient import TestClient
    from pyinstrument import Profiler

    from app.main import app
    from tests.helpers import login, submit_complete

    latencies = []
    with TestClient(app) as client:
        headers = login(client, "basvuran")
        submit_complete(client, headers, "temiz")  # warm-up (models, SHAP, fonts)
        profiler = Profiler()
        for run in range(args.runs):
            last = run == args.runs - 1
            if last:
                profiler.start()
            started = time.perf_counter()
            app_id = submit_complete(client, headers, "temiz")
            latencies.append(time.perf_counter() - started)
            if last:
                profiler.stop()
            state = client.get(f"/api/v1/applications/{app_id}", headers=headers).json()["state"]
            print(f"run {run + 1}: {latencies[-1]:.2f}s -> {state}")
    print(f"p50 {statistics.median(latencies):.2f}s  max {max(latencies):.2f}s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(profiler.output_text(unicode=True, color=False, show_all=False)[:12000])
    if args.html:
        Path(args.html).write_text(profiler.output_html(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
