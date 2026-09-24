"""Generate synthetic Turkish sample documents (clean and tampered sets).

Writes ``data/generated/samples/<persona>/`` bundles: identity, payslip,
SGK service record, e-Devlet residence document (QR + barcode) and bank
statement. The ``kurcalanmis`` set contains a tampered payslip (incremental
save, editing-tool producer, foreign font, broken arithmetic) and a bank
statement whose closing balance does not add up.

Usage: python scripts/generate_sample_docs.py [--out data/generated/samples]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.documents.samples import generate_applicant_bundle
from app.integrations.personas import DEMO_TCKN, open_banking_transactions

PEOPLE = {
    "temiz": ("Elif Yıldırım", 45_000),
    "ince_dosya": ("Kerem Aydın", 35_000),
    "kurcalanmis": ("Onur Çelik", 45_000),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/generated/samples")
    args = parser.parse_args()
    for persona, (name, income) in PEOPLE.items():
        tckn = DEMO_TCKN[persona]
        ob = open_banking_transactions(tckn, income)
        tampered = persona == "kurcalanmis"
        paths = generate_applicant_bundle(
            Path(args.out) / persona,
            name=name,
            tckn=tckn,
            iban="TR330006100519786457841326",
            address="Moda Cad. No:5 Kadıköy İstanbul",
            employer="Anadolu Bilişim Ltd. Şti.",
            net_income=income,
            transactions=ob["transactions"],
            opening_balance=ob["account"]["opening_balance"],
            tamper_payslip=tampered,
            tamper_statement=tampered,
        )
        print(persona, {code: str(path) for code, path in paths.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
