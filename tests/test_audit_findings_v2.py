"""Regression tests for the v2 independent-audit findings (one test per finding).

Each test was written *before* its fix and marked ``xfail(strict=True)`` so
the reproduction was recorded red; the fix commit removes the marker.
Finding numbers follow ``docs/PLAN_v2.md``.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT
from app.core.rules import load_policy_file, load_pricing
from app.decisioning.engine import decide
from app.decisioning.models import get_models
from tests.test_decisioning import _snapshot

V0 = "V0 reproduction: audit finding not fixed yet"


def _decide(persona: str, income: float, amount: float, term: int = 36):
    snapshot = _snapshot(persona, income, amount, term)
    return snapshot, decide(
        snapshot, policy=load_policy_file(), models=get_models(), pricing_cfg=load_pricing()
    )


def _git_grep(pattern: str) -> list[str]:
    if shutil.which("git") is None or not (PROJECT_ROOT / ".git").exists():
        pytest.skip("git repository not available")
    result = subprocess.run(
        ["git", "grep", "-n", "-I", pattern, "--", ".", ":!tests/test_audit_findings_v2.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


# ------------------------------------------------------------------ A. decisioning / pricing
@pytest.mark.xfail(strict=True, reason=V0)
def test_f01_offer_dsr_rechecked_with_priced_taxed_instalment():
    """Audit case: 800k / 36 m for a 45k earner; the priced instalment breached the DSR cap."""
    snapshot, result = _decide("temiz", 45_000, 800_000)
    assert result.pricing is not None
    real_dsr = (snapshot["existing_debt_service"] + result.pricing.instalment) / snapshot[
        "monthly_income"
    ]
    assert real_dsr <= load_policy_file().product("IHTIYAC").max_dsr + 1e-6
    assert result.limits["dsr_offer"] == pytest.approx(real_dsr, abs=1e-3)


@pytest.mark.xfail(strict=True, reason=V0)
def test_f02_non_material_tenure_does_not_produce_reason():
    """52 months of employment on a PD≈1% file must not read as 'short tenure'."""
    snapshot, result = _decide("temiz", 45_000, 800_000)
    assert snapshot["employment_months"] >= 48
    assert "R05_ISTIHDAM_KISA" not in result.reason_code_list
    assert len(result.reason_codes) <= 4
    model_reasons = [r for r in result.reason_codes if r.source == "model"]
    assert all(r.kind == "improvement" for r in model_reasons)


@pytest.mark.xfail(strict=True, reason=V0)
def test_f03_basel_other_retail_correlation():
    from app.pricing.engine import retail_correlation

    def expected(pd: float) -> float:
        w = (1 - math.exp(-35 * pd)) / (1 - math.exp(-35))
        return 0.03 * w + 0.16 * (1 - w)

    for pd in (0.0003, 0.01, 0.05, 0.2):
        assert retail_correlation(pd) == pytest.approx(expected(pd), rel=1e-9)
    assert retail_correlation(0.01) == pytest.approx(0.12161, abs=1e-4)


@pytest.mark.xfail(strict=True, reason=V0)
def test_f04_user_facing_labels_for_enum_codes():
    from app.core.labels import label

    assert label("product", "IHTIYAC") == "İhtiyaç Kredisi"
    assert label("employment_type", "MAASLI") == "Maaşlı çalışan"
    assert label("product", "UNKNOWN_CODE") == "UNKNOWN_CODE"


# ------------------------------------------------------------------ B. documents
@pytest.mark.xfail(strict=True, reason=V0)
def test_f05_genuine_document_from_other_tool_is_not_flagged(tmp_path):
    """A genuine e-Devlet PDF produced by a non-reportlab tool must not score as fraud."""
    import pikepdf

    from app.documents.fraud import metadata_signals
    from app.documents.samples import make_edevlet_document

    path = tmp_path / "adres.pdf"
    make_edevlet_document(
        path,
        kind="YERLESIM",
        name="Ayşe Kaya",
        tckn="10000000146",
        detail="Moda Cad. No:5 Kadıköy İstanbul",
    )
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        pdf.docinfo["/Producer"] = "iText® 7.1.16 ©2000-2021 iText Group NV"
        pdf.docinfo["/Creator"] = "e-Devlet Kapısı"
        pdf.save(path)
    codes = {s.code for s in metadata_signals(path, "ADDRESS")}
    assert "producer_mismatch" not in codes


@pytest.mark.xfail(strict=True, reason=V0)
def test_f06_ocr_status_reported_in_health():
    from fastapi.testclient import TestClient

    from app.main import app

    body = TestClient(app).get("/health").json()
    assert "ocr" in body and "available" in body["ocr"]


# ------------------------------------------------------------------ C. governance
def test_f07_lda_table_compares_models_at_same_approval_rate():
    from app.decisioning.features import MODEL_FEATURES
    from app.decisioning.training import generate_dataset
    from app.governance.fairness import less_discriminatory_alternatives

    df = generate_dataset(6000, seed=7)
    rows = less_discriminatory_alternatives(df, list(MODEL_FEATURES), "gender")
    rates = [r["approval_rate"] for r in rows]
    assert max(rates) - min(rates) <= 0.01


@pytest.mark.xfail(strict=True, reason=V0)
def test_f07_synthetic_proxies_correlate_with_protected_attributes():
    from app.decisioning.training import generate_dataset

    df = generate_dataset(8000, seed=11)
    assert df["age"].corr(df["employment_months"]) > 0.2


def test_f08_champion_challenger_shows_statistical_evidence():
    from app.db.session import init_db, session_scope
    from app.governance.inventory import champion_challenger

    init_db()
    with session_scope() as session:
        data = champion_challenger(session)
    evidence = data["validation"]
    assert "delong_p_value" in evidence and "auc_ci" in evidence


# ------------------------------------------------------------------ D. concurrency
@pytest.mark.xfail(strict=True, reason=V0)
def test_f09_sqlite_uses_wal_and_normal_sync():
    from sqlalchemy import text

    from app.db.session import get_engine

    with get_engine().connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA synchronous")).scalar() == 1  # NORMAL


@pytest.mark.xfail(strict=True, reason=V0)
def test_f10_half_open_admits_a_single_probe():
    import fakeredis

    from app.integrations.circuit_breaker import CircuitBreaker, RedisBreakerStore

    store = RedisBreakerStore(fakeredis.FakeRedis())
    breaker = CircuitBreaker("probe", failure_threshold=1, reset_timeout=0.0, store=store)
    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert breaker._allow_request() is True  # the probe
    assert breaker._allow_request() is False  # everyone else waits for the probe


@pytest.mark.xfail(strict=True, reason=V0)
def test_f11_single_pending_review_per_application():
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    from app.db.models import Review
    from app.db.session import init_db, session_factory

    init_db()
    session = session_factory()()
    # Isolate the uniqueness rule from the foreign key to applications.
    session.execute(text("PRAGMA foreign_keys=OFF"))
    try:
        for _ in range(2):
            session.add(
                Review(
                    application_id="APP-UNIQ-TEST",
                    maker="u",
                    maker_role="uzman",
                    action="ONAY",
                    justification="x",
                    required_role="uzman",
                    status="ONAY_BEKLIYOR",
                )
            )
        with pytest.raises(IntegrityError):
            session.flush()
    finally:
        session.rollback()
        session.execute(text("PRAGMA foreign_keys=ON"))
        session.close()


# ------------------------------------------------------------------ E. security
@pytest.mark.xfail(strict=True, reason=V0)
def test_f13_csp_without_unsafe_eval():
    from fastapi.testclient import TestClient

    from app.main import app

    csp = TestClient(app).get("/health").headers["Content-Security-Policy"]
    assert "unsafe-eval" not in csp


def test_f14_login_sets_httponly_cookie_and_csrf_is_enforced():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/login", json={"username": "basvuran", "password": "Demo123!"}
        )
        cookie = response.headers.get("set-cookie", "")
        assert "HttpOnly" in cookie and "SameSite=strict" in cookie.replace("Strict", "strict")
        blocked = client.post("/api/v1/applications", json={})  # cookie, no CSRF header
        assert blocked.status_code == 403


def test_f15_prod_defaults_disable_demo_users():
    from app.core.config import Settings

    settings = Settings(_env_file=None, app_env="prod")  # type: ignore[call-arg]
    assert settings.seed_demo_users is False
    assert Settings(_env_file=None, app_env="dev").seed_demo_users is True  # type: ignore[call-arg]


def test_f16_redaction_handles_turkish_dotted_i():
    from app.agents.redaction import Redactor

    # Bank/core systems often ASCII-fold names ("ISMAIL ISIK"); the surname leaked.
    for registered in ("İsmail Işık", "ISMAIL ISIK"):
        redactor = Redactor.for_applicant(name=registered)
        for variant in ("İSMAİL IŞIK", "ismail ışık", "Ismail Işık", "ISMAIL ISIK", "ismail isik"):
            out = redactor.redact(f"Müşteri {variant} başvurdu")
            assert out == "Müşteri BASVURAN_1 başvurdu", (registered, variant, out)


# ------------------------------------------------------------------ F. UI
@pytest.mark.xfail(strict=True, reason=V0)
def test_f17_queue_is_paginated():
    from fastapi.testclient import TestClient

    from app.main import app
    from tests.helpers import login

    with TestClient(app) as client:
        body = client.get(
            "/api/v1/workbench/queue?limit=5&offset=0", headers=login(client, "uzman")
        ).json()
    assert {"total", "limit", "offset"} <= set(body)
    assert len(body["items"]) <= 5


@pytest.mark.xfail(strict=True, reason=V0)
def test_f18_every_form_control_is_labelled():
    html = (PROJECT_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    controls = re.findall(r"<(input|select|textarea)\b([^>]*)>", html)
    assert controls
    for tag, attrs in controls:
        if 'type="hidden"' in attrs:
            continue
        control_id = re.search(r'\bid="([^"]+)"', attrs)
        labelled = "aria-label" in attrs or "aria-labelledby" in attrs
        if control_id:
            labelled = labelled or f'for="{control_id.group(1)}"' in html
        assert labelled, f"<{tag}{attrs[:80]}> has no label"


# ------------------------------------------------------------------ G. honesty
@pytest.mark.xfail(strict=True, reason=V0)
def test_f20_no_unverified_sr_26_2_citation():
    assert _git_grep("SR 26-2") == []


@pytest.mark.xfail(strict=True, reason=V0)
def test_f21_readme_has_limitations_instead_of_vendor_parity_table():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Limitations" in readme
    assert "## Inspired-by patterns" in readme
    assert "Ocrolus" not in readme.split("## Inspired-by patterns")[0]


@pytest.mark.xfail(strict=True, reason=V0)
def test_f22_internal_llm_host_not_in_tracked_files():
    assert _git_grep("llm-gateway.example.org") == []


@pytest.mark.xfail(strict=True, reason=V0)
def test_f23_final_report_v2_exists():
    report = PROJECT_ROOT / "docs" / "FINAL_REPORT_v2.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    for section in ("Findings", "Lane A", "Champion", "Fairness", "Performance", "Audit rounds"):
        assert section in text, section


def test_fixture_is_small_and_attributed():
    fixture = Path(__file__).parent / "fixtures" / "uci_taiwan_sample.csv"
    assert fixture.stat().st_size < 1_000_000
    head = fixture.read_text(encoding="utf-8").splitlines()[:3]
    assert any("CC BY 4.0" in line for line in head)
