"""Run the real application in a uvicorn subprocess for browser tests.

Each server gets a free port and its own temporary SQLite database and
artifact directories; :func:`stop_server` terminates only that process.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_env(tmp: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_", "DATABASE_"))}
    env.update(
        {
            "ANIL2_NO_DOTENV": "1",
            "APP_ENV": "test",
            "LLM_MODE": "demo",
            "LLM_API_KEY": "",
            "TASK_QUEUE_BACKEND": "inline",
            "CIRCUIT_STATE_BACKEND": "memory",
            "RATE_LIMIT_ENABLED": "false",
            "COOKIE_SECURE": "false",
            "DATABASE_URL": f"sqlite:///{(tmp / 'e2e.db').as_posix()}",
            "REPORT_OUTPUT_DIR": str(tmp / "reports"),
            "RESULT_STORE_DIR": str(tmp / "results"),
            "UPLOAD_DIR": str(tmp / "uploads"),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return env


def start_server(tmp: Path, timeout: float = 120.0) -> tuple[subprocess.Popen, str]:
    """Start uvicorn on a free port and wait until ``/health/ready`` answers 200."""
    tmp.mkdir(parents=True, exist_ok=True)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(tmp / "server.log", "wb")  # noqa: SIM115 - closed by stop_server
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        env=server_env(tmp),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    proc.log_file = log  # type: ignore[attr-defined]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.close()
            output = (tmp / "server.log").read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"uvicorn exited with {proc.returncode}:\n{output[-4000:]}")
        try:
            with urllib.request.urlopen(base + "/health/ready", timeout=2) as response:
                if response.status == 200:
                    return proc, base
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    stop_server(proc)
    raise RuntimeError(f"server on {base} did not become ready in {timeout:.0f}s")


def stop_server(proc: subprocess.Popen) -> None:
    """Terminate only the process this module started."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
    log = getattr(proc, "log_file", None)
    if log is not None and not log.closed:
        log.close()
