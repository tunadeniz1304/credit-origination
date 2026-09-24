"""Persistence tests: state machine, audit hash chain, outbox, crypto, JWT."""

from __future__ import annotations

from itertools import pairwise

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session, sessionmaker

from app.core import crypto
from app.core.config import Settings
from app.core.security import (
    AuthError,
    Principal,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.audit import append_audit, verify_chain
from app.db.models import AuditLog, Base, OutboxMessage
from app.db.outbox import ConsoleAdapter, enqueue
from app.workflow.states import (
    TERMINAL,
    TRANSITIONS,
    InvalidTransitionError,
    State,
    assert_transition,
    can_transition,
)


@pytest.fixture()
def session(tmp_path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'p.db'}")
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)
    with maker() as s:
        yield s


# ------------------------------------------------------------------ state machine
def test_happy_path_transitions_are_allowed():
    path = [
        State.TASLAK,
        State.GONDERILDI,
        State.BELGE_BEKLENIYOR,
        State.BELGE_INCELEMEDE,
        State.VERI_TOPLANIYOR,
        State.KARAR_MOTORU,
        State.OTOMATIK_ONAY,
        State.TEKLIF_SUNULDU,
        State.TEKLIF_KABUL,
        State.SOZLESME_HAZIR,
        State.KULLANDIRILDI,
    ]
    for current, target in pairwise(path):
        assert can_transition(current, target), (current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (State.TASLAK, State.KULLANDIRILDI),
        (State.OTOMATIK_RET, State.TEKLIF_SUNULDU),
        (State.KULLANDIRILDI, State.IPTAL),
        (State.IPTAL, State.GONDERILDI),
        (State.KARAR_MOTORU, State.TEKLIF_SUNULDU),
        (State.TEKLIF_SUNULDU, State.KULLANDIRILDI),
    ],
)
def test_invalid_transitions_rejected(current, target):
    with pytest.raises(InvalidTransitionError):
        assert_transition(current, target)


def test_objection_loop_and_terminal_states():
    assert can_transition(State.OTOMATIK_RET, State.ITIRAZ_INCELEMESI)
    assert can_transition(State.ITIRAZ_INCELEMESI, State.TEKLIF_SUNULDU)
    assert {State.KULLANDIRILDI, State.IPTAL} == TERMINAL
    assert set(TRANSITIONS) == set(State)


# ------------------------------------------------------------------ audit chain
def test_audit_chain_verifies_and_detects_tampering(session):
    for index in range(5):
        append_audit(
            session,
            actor="uzman1",
            action="TEST",
            entity_type="application",
            entity_id=f"APP-{index}",
            payload={"i": index},
        )
    session.commit()
    assert verify_chain(session) == {**verify_chain(session), "valid": True, "entries": 5}
    session.execute(update(AuditLog).where(AuditLog.id == 3).values(payload={"i": 999}))
    session.commit()
    report = verify_chain(session)
    assert report["valid"] is False and report["broken_at"] == 3


def test_audit_chain_links_prev_hash(session):
    first = append_audit(session, actor="a", action="X", entity_type="t", entity_id="1")
    second = append_audit(session, actor="a", action="Y", entity_type="t", entity_id="1")
    session.commit()
    assert second.prev_hash == first.hash and first.prev_hash == "0" * 64


# ------------------------------------------------------------------ outbox
def test_outbox_enqueue_is_idempotent(session):
    first = enqueue(session, event="E", aggregate_id="APP-1", payload={}, idempotency_key="k1")
    duplicate = enqueue(session, event="E", aggregate_id="APP-1", payload={}, idempotency_key="k1")
    session.commit()
    assert first is not None and duplicate is None
    assert session.query(OutboxMessage).count() == 1


def test_console_adapter_records_key():
    adapter = ConsoleAdapter()
    adapter.send(
        OutboxMessage(
            event="E", aggregate_id="A", idempotency_key="k", channel="console", payload={}
        )
    )
    assert adapter.sent == ["k"]


# ------------------------------------------------------------------ crypto
def test_encrypt_round_trip_and_blind_index_stability():
    settings = Settings(_env_file=None)
    token = crypto.encrypt("10000000146", settings)
    assert token and "10000000146" not in token
    assert crypto.decrypt(token, settings) == "10000000146"
    assert crypto.blind_index("TR33 0006 1005", settings) == crypto.blind_index(
        "tr3300061005", settings
    )
    assert crypto.blind_index("a", settings) != crypto.blind_index("b", settings)
    assert crypto.decrypt("not-a-token", settings) is None


def test_masks():
    assert crypto.mask_tckn("10000000146") == "10*******46"
    assert crypto.mask_iban("TR330006100519786457841326").endswith("1326")
    assert crypto.mask_phone("05321234567").endswith("67")


# ------------------------------------------------------------------ security
def test_password_hashing():
    encoded = hash_password("Gizli123!", iterations=1000)
    assert verify_password("Gizli123!", encoded)
    assert not verify_password("yanlış", encoded)
    assert not verify_password("x", "garbage")


def test_jwt_round_trip_and_tamper_detection():
    settings = Settings(_env_file=None, jwt_secret="unit-test-signing-material-0123456789")
    principal = Principal(user_id="u1", username="uzman1", role="uzman", full_name="Uzman Bir")
    token = create_access_token(principal, settings)
    decoded = decode_token(token, settings)
    assert decoded.role == "uzman" and decoded.is_staff and decoded.authority_rank == 1
    with pytest.raises(AuthError):
        decode_token(token + "x", settings)
    other = Settings(_env_file=None, jwt_secret="another-signing-material-9876543210")
    with pytest.raises(AuthError):
        decode_token(token, other)
