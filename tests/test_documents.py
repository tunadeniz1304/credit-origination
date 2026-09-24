"""IDP tests: storage safety, extraction, tamper signals, synthetic documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.documents.extraction import extract_fields, extract_text, parse_amount
from app.documents.fraud import analyse_document
from app.documents.samples import (
    generate_applicant_bundle,
    make_barcode,
    make_edevlet_document,
    verify_barcode,
)
from app.documents.storage import UploadRejected, detect_mime, sanitise_filename, store_upload
from app.integrations.personas import DEMO_TCKN, open_banking_transactions

NAME = "Ayşe Kaya"


@pytest.fixture(scope="module")
def bundles(tmp_path_factory) -> dict[str, dict[str, Path]]:
    tckn = DEMO_TCKN["kurcalanmis"]
    ob = open_banking_transactions(tckn, 45_000)
    root = tmp_path_factory.mktemp("docs")
    common = dict(
        name=NAME,
        tckn=tckn,
        iban="TR330006100519786457841326",
        address="Moda Cad. No:5 Kadıköy",
        employer="Anadolu Bilişim Ltd. Şti.",
        net_income=45_000,
        transactions=ob["transactions"],
        opening_balance=ob["account"]["opening_balance"],
    )
    return {
        "clean": generate_applicant_bundle(root / "clean", **common),
        "tampered": generate_applicant_bundle(
            root / "tampered", tamper_payslip=True, tamper_statement=True, **common
        ),
    }


def _analyse(path: Path, code: str):
    text = extract_text(path, "application/pdf")
    fields = extract_fields(code, text, path)
    return text, fields, analyse_document(path, code, "application/pdf", fields, text)


def test_clean_documents_have_no_fraud_signals(bundles):
    for code, path in bundles["clean"].items():
        _, _, report = _analyse(path, code)
        assert report.score == 0.0, (code, report.signals)


def test_tampered_payslip_detected(bundles):
    _, fields, report = _analyse(bundles["tampered"]["INCOME"], "INCOME")
    codes = {s.code for s in report.signals}
    assert {
        "incremental_update",
        "editing_tool",
        "arithmetic_mismatch",
        "font_inconsistency",
    } <= codes
    assert report.score >= 0.6
    net = next(f for f in fields if f.name == "net_ucret")
    assert parse_amount(net.value) > 60_000  # the inflated value is what was extracted


def test_tampered_statement_arithmetic(bundles):
    _, _, report = _analyse(bundles["tampered"]["BANK_STATEMENT"], "BANK_STATEMENT")
    assert [s.code for s in report.signals] == ["arithmetic_mismatch"]


def test_payslip_fields_with_confidence_and_bbox(bundles):
    _, fields, _ = _analyse(bundles["clean"]["INCOME"], "INCOME")
    by_name = {f.name: f for f in fields}
    assert by_name["tckn"].value == DEMO_TCKN["kurcalanmis"]
    assert by_name["ad_soyad"].value == NAME  # Turkish glyphs survive extraction
    assert by_name["net_ucret"].confidence >= 0.9
    assert by_name["net_ucret"].page == 1 and by_name["net_ucret"].bbox


def test_barcode_checksum_and_invalid_barcode(tmp_path):
    good = make_barcode("seed")
    assert verify_barcode(good)
    assert not verify_barcode(good[:-2] + "00" if good[-2:] != "00" else good[:-2] + "01")
    forged = tmp_path / "forged.pdf"
    make_edevlet_document(
        forged,
        kind="YERLESIM",
        name=NAME,
        tckn="10000000146",
        detail="X",
        barcode="ED00000000000099",
    )
    _, _, report = _analyse(forged, "ADDRESS")
    assert "barcode_invalid" in {s.code for s in report.signals}


def test_detect_mime_by_magic_bytes():
    assert detect_mime(b"%PDF-1.7 ...") == "application/pdf"
    assert detect_mime(b"\x89PNG\r\n\x1a\n....") == "image/png"
    assert detect_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert detect_mime("düz metin".encode()) == "text/plain"
    assert detect_mime(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff") is None


def test_sanitise_filename_blocks_traversal():
    assert sanitise_filename("../../etc/passwd") == "passwd"
    assert sanitise_filename("bordro ağustos.pdf") == "bordro_a_ustos.pdf"
    assert sanitise_filename("") == "belge"


def test_store_upload_isolates_per_application(tmp_path):
    settings = Settings(_env_file=None, upload_dir=str(tmp_path))
    a = store_upload("APP-AAAAAAAAAAAA", "INCOME", "bordro.pdf", b"%PDF-1.4 a", settings)
    b = store_upload("APP-BBBBBBBBBBBB", "INCOME", "bordro.pdf", b"%PDF-1.4 b", settings)
    assert Path(a.path).parent.name == "APP-AAAAAAAAAAAA"
    assert Path(b.path).parent.name == "APP-BBBBBBBBBBBB"
    assert Path(a.path).read_bytes() == b"%PDF-1.4 a"  # not overwritten by B (bug #1)
    assert a.sha256 != b.sha256


@pytest.mark.parametrize(
    ("app_id", "data", "message"),
    [
        ("APP-AAAAAAAAAAAA", b"", "empty"),
        ("APP-AAAAAAAAAAAA", b"MZ\x90\x00binary\x00\x00", "unsupported"),
        ("../../etc", b"%PDF-1.4", "invalid application id"),
    ],
)
def test_store_upload_rejections(tmp_path, app_id, data, message):
    settings = Settings(_env_file=None, upload_dir=str(tmp_path))
    with pytest.raises(UploadRejected, match=message):
        store_upload(app_id, "INCOME", "x.pdf", data, settings)


def test_store_upload_size_limit(tmp_path):
    settings = Settings(_env_file=None, upload_dir=str(tmp_path), max_upload_mb=0.001)
    with pytest.raises(UploadRejected, match="limit"):
        store_upload("APP-AAAAAAAAAAAA", "INCOME", "x.pdf", b"%PDF-" + b"0" * 5000, settings)


def test_text_extraction_for_plain_text(tmp_path):
    path = tmp_path / "belge.txt"
    path.write_text("Net Ücret: 12.345,67 TL", encoding="utf-8")
    result = extract_text(path, "text/plain")
    assert result.source == "digital"
    assert extract_fields("INCOME", result)[0].value == "12.345,67"


def test_image_without_ocr_requires_manual_review(tmp_path):
    path = tmp_path / "scan.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    result = extract_text(path, "image/png")
    assert result.source in ("none", "ocr")
