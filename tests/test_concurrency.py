"""Concurrency against a real server process (audit F09).

Starts ``uvicorn`` on a free port with a fresh SQLite database in inline mode,
then fires 20 concurrent applications (each uploading its five documents, so
the pipeline runs end to end) together with 20 concurrent logins. Expected:
zero 5xx responses and an intact audit hash chain.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from app.core.config import PROJECT_ROOT
from app.integrations import personas
from app.kyc.tckn import synthetic_tckn
from tests.helpers import DEMO_LOGIN, bundle

APPLICATIONS = 20
LOGINS = 20
TERMINAL = {"TEKLIF_SUNULDU", "UZMAN_INCELEMESI", "OTOMATIK_RET", "BELGE_BEKLENIYOR"}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def server():
    tmp = Path(tempfile.mkdtemp(prefix="anil2-concurrency-"))
    port = _free_port()
    log = open(tmp / "server.log", "wb")  # noqa: SIM115 - closed after the server stops
    env = {
        **os.environ,
        "ANIL2_NO_DOTENV": "1",
        "APP_ENV": "test",
        "LLM_MODE": "demo",
        "LLM_API_KEY": "",
        "TASK_QUEUE_BACKEND": "inline",
        "CIRCUIT_STATE_BACKEND": "memory",
        "RATE_LIMIT_ENABLED": "false",
        "COOKIE_SECURE": "false",
        "DATABASE_URL": f"sqlite:///{(tmp / 'c.db').as_posix()}",
        "REPORT_OUTPUT_DIR": str(tmp / "reports"),
        "RESULT_STORE_DIR": str(tmp / "results"),
        "UPLOAD_DIR": str(tmp / "uploads"),
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        # A file, never an unread PIPE: a full pipe buffer would block the server's logging.
        stdout=subprocess.DEVNULL,
        stderr=log,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/health/ready", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail("server did not start")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.kill()
    log.close()


def _token(base: str, username: str) -> dict[str, str]:
    response = httpx.post(
        f"{base}/api/v1/auth/login",
        json={"username": username, "password": DEMO_LOGIN},
        timeout=60,
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _apply(base: str, headers: dict[str, str], index: int) -> list[int]:
    statuses: list[int] = []
    persona = ("temiz", "gri", "ince_dosya", "yuksek_dsr")[index % 4]
    tckn = synthetic_tckn(f"concurrency-{index}-{time.time()}")
    personas._PINNED[tckn] = persona  # only affects this process; the server maps by hash
    income = 30_000 + (index % 5) * 5_000
    body = {
        "name": f"Eşzamanlı Test {index}",
        "identity_no": tckn,
        "birth_date": "1988-04-12",
        "phone": f"0549{index:07d}",
        "email": "c@example.com",
        "address": f"Test Sok. No:{index}",
        "iban": f"TR55000610000000{index:010d}",
        "monthly_income": income,
        "requested_amount": income * 4,
        "requested_term_months": 36,
        "consents": {
            "kvkk_aydinlatma": True,
            "acik_riza": True,
            "kkb_sorgu": True,
            "edevlet_sorgu": True,
            "acik_bankacilik": True,
        },
    }
    with httpx.Client(base_url=base, headers=headers, timeout=120) as client:
        created = client.post("/api/v1/applications", json=body)
        statuses.append(created.status_code)
        if created.status_code != 202:
            return statuses
        app_id = created.json()["application_id"]
        for code, path in bundle(persona, income, False, body["name"], body["iban"], tckn).items():
            with open(path, "rb") as fh:
                response = client.post(
                    f"/api/v1/applications/{app_id}/documents",
                    data={"code": code},
                    files={"file": (path.name, fh, "application/pdf")},
                )
            statuses.append(response.status_code)
        deadline = time.time() + 150
        while time.time() < deadline:
            detail = client.get(f"/api/v1/applications/{app_id}")
            statuses.append(detail.status_code)
            if detail.status_code == 200 and detail.json()["state"] in TERMINAL - {
                "BELGE_BEKLENIYOR"
            }:
                break
            time.sleep(1)
    return statuses


def _login(base: str, index: int) -> int:
    user = ("uzman", "basvuran", "kidemli", "modelyon")[index % 4]
    return httpx.post(
        f"{base}/api/v1/auth/login",
        json={"username": user, "password": DEMO_LOGIN},
        timeout=60,
    ).status_code


def test_twenty_applications_and_twenty_logins_without_server_errors(server):
    headers = _token(server, "basvuran")
    with ThreadPoolExecutor(max_workers=APPLICATIONS + LOGINS) as pool:
        apps = [pool.submit(_apply, server, headers, i) for i in range(APPLICATIONS)]
        logins = [pool.submit(_login, server, i) for i in range(LOGINS)]
        statuses = [s for f in apps for s in f.result()] + [f.result() for f in logins]
    server_errors = [s for s in statuses if s >= 500]
    assert server_errors == [], f"{len(server_errors)} server errors out of {len(statuses)}"
    assert statuses.count(200) >= LOGINS
    admin = _token(server, "admin")
    chain = httpx.get(f"{server}/api/v1/audit/verify", headers=admin, timeout=60).json()
    assert chain["valid"] is True and chain["entries"] > APPLICATIONS * 5
