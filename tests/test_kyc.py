"""KYC / fraud pre-check tests: TCKN checksum, sanctions, rings, anomaly."""

from __future__ import annotations

import pytest

from app.kyc.anomaly import anomaly_score
from app.kyc.network import IdentityRecord, ring_for
from app.kyc.sanctions import normalise_name, screen_name
from app.kyc.tckn import complete_tckn, is_valid_tckn, synthetic_tckn


@pytest.mark.parametrize("value", ["10000000146", "11111111110", synthetic_tckn("x")])
def test_valid_tckn(value):
    assert is_valid_tckn(value)


@pytest.mark.parametrize(
    "value",
    ["12345678901", "01234567890", "1000000014", "1000000014a", "10000000147", "34567890123"],
)
def test_invalid_tckn(value):
    assert not is_valid_tckn(value)


def test_complete_tckn_round_trip():
    assert complete_tckn("100000001") == "10000000146"
    with pytest.raises(ValueError):
        complete_tckn("012345678")


def test_synthetic_tckn_is_deterministic():
    assert synthetic_tckn("seed") == synthetic_tckn("seed")
    assert synthetic_tckn("seed") != synthetic_tckn("other")


def test_turkish_normalisation():
    assert normalise_name("  Gülten   ARIKBOĞA ") == "gulten arikboga"
    assert normalise_name("İSMAİL IŞIK") == "ismail isik"


def test_sanctions_exact_and_fuzzy_match():
    exact = screen_name("Kasım Yılmazer")
    assert exact.matched and exact.hits[0].list == "DEMO-BM-1267"
    fuzzy = screen_name("Kasim Yilmazer")  # diacritics dropped
    assert fuzzy.matched
    reordered = screen_name("Yılmazer Kasım")
    assert reordered.matched


def test_sanctions_no_false_positive_for_common_name():
    result = screen_name("Ayşe Kaya")
    assert not result.matched
    assert result.best_score < 0.92


def _rec(
    app_id: str, applicant: str, phone: str | None = None, iban: str | None = None
) -> IdentityRecord:
    return IdentityRecord(
        app_id, applicant, {"phone": phone, "iban": iban, "device": None, "address": None}
    )


def test_ring_detection_flags_three_applicants_sharing_identifiers():
    records = [
        _rec("A1", "p1", phone="ph1"),
        _rec("A2", "p2", phone="ph1", iban="ib1"),
        _rec("A3", "p3", iban="ib1"),
        _rec("A4", "p4", phone="ph9"),
    ]
    ring = ring_for("A3", records, min_size=3)
    assert ring.flagged and ring.ring_size == 3
    assert set(ring.shared_identifiers) == {"phone", "iban"}
    assert any(n.kind == "application" for n in ring.nodes) and ring.edges
    alone = ring_for("A4", records, min_size=3)
    assert not alone.flagged and alone.ring_size == 1


def test_same_applicant_reapplying_is_not_a_ring():
    records = [
        _rec("A1", "p1", phone="x"),
        _rec("A2", "p1", phone="x"),
        _rec("A3", "p1", phone="x"),
    ]
    assert not ring_for("A1", records).flagged


def test_unknown_application_has_no_ring():
    assert ring_for("ZZZ", []).ring_size == 1


def test_anomaly_score_bounds_and_ordering():
    normal = anomaly_score(income=42_000, amount=150_000, term_months=36, age=35)
    odd = anomaly_score(income=17_000, amount=1_500_000, term_months=6, age=22)
    assert 0.0 <= normal <= 1.0 and 0.0 <= odd <= 1.0
    assert odd > normal
