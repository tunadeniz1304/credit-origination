"""Download public credit-default datasets used to validate the methodology.

This is the **only** component of the project that goes to the network. It
downloads each dataset, verifies the SHA-256 of the archive against the pinned
value below, normalises it to CSV under ``data/external/<set>/`` (gitignored)
and writes ``data/external/manifest.json``. It can also (re)build the small
stratified fixture that is committed for offline tests.

Datasets
--------
* ``uci_taiwan``  - UCI "Default of Credit Card Clients" (Yeh & Lien, 2009),
  30,000 rows, CC BY 4.0, https://doi.org/10.24432/C55S3H
* ``german_credit`` - UCI "Statlog (German Credit Data)" (Hofmann, 1994),
  1,000 rows, CC BY 4.0, https://doi.org/10.24432/C5NC77
* Kaggle sets (Home Credit, Give Me Some Credit, Lending Club) are used only
  when ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` exist in the environment; the
  script checks for their presence only and never reads or prints the values.

Usage::

    python scripts/fetch_public_credit_data.py [--sets uci_taiwan german_credit]
                                               [--fixture] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

EXTERNAL = ROOT / "data" / "external"
FIXTURE = ROOT / "tests" / "fixtures" / "uci_taiwan_sample.csv"
FIXTURE_ROWS = 3000
FIXTURE_SEED = 20260925
USER_AGENT = "anil2-credit-platform/2.1 (research; +https://github.com/tunadeniz1304/Anil2)"

DATASETS: dict[str, dict[str, str]] = {
    "uci_taiwan": {
        "title": "Default of Credit Card Clients (UCI id 350)",
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "sha256": "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602",
        "license": "CC BY 4.0",
        "citation": (
            "Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C55S3H"
        ),
    },
    "german_credit": {
        "title": "Statlog (German Credit Data) (UCI id 144)",
        "url": "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
        "sha256": "e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b",
        "license": "CC BY 4.0",
        "citation": (
            "Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C5NC77"
        ),
    },
}
KAGGLE_SETS = ("home-credit-default-risk", "GiveMeSomeCredit", "lending-club")

GERMAN_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "class",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data: bytes = response.read()
    return data


def normalise_taiwan(archive: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xls"))
        frame = pd.read_excel(io.BytesIO(zf.read(name)), header=1)
    frame = frame.rename(columns={"default payment next month": "default"})
    return frame.drop(columns=["ID"])


def normalise_german(archive: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        raw = zf.read("german.data").decode("ascii")
    frame = pd.read_csv(io.StringIO(raw), sep=" ", header=None, names=GERMAN_COLUMNS)
    frame["default"] = (frame.pop("class") == 2).astype(int)  # 1 = good, 2 = bad
    return frame


NORMALISERS = {"uci_taiwan": normalise_taiwan, "german_credit": normalise_german}


def fetch(name: str, force: bool = False) -> dict[str, object]:
    spec = DATASETS[name]
    target = EXTERNAL / name
    target.mkdir(parents=True, exist_ok=True)
    archive_path = target / "source.zip"
    if archive_path.is_file() and not force:
        archive = archive_path.read_bytes()
    else:
        archive = download(spec["url"])
    digest = sha256(archive)
    if digest != spec["sha256"]:
        raise SystemExit(
            f"{name}: checksum mismatch (expected {spec['sha256']}, got {digest}); "
            "the upstream file changed, verify it before updating the pinned value"
        )
    archive_path.write_bytes(archive)
    frame = NORMALISERS[name](archive)
    csv_path = target / "data.csv"
    frame.to_csv(csv_path, index=False)
    return {
        "title": spec["title"],
        "url": spec["url"],
        "license": spec["license"],
        "citation": spec["citation"],
        "archive_sha256": digest,
        "csv_sha256": sha256(csv_path.read_bytes()),
        "rows": len(frame),
        "columns": len(frame.columns),
        "default_rate": round(float(frame["default"].mean()), 4),
        "path": str(csv_path.relative_to(ROOT).as_posix()),
    }


def kaggle_status() -> dict[str, str]:
    present = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    if not present:
        return {s: "atlandı (KAGGLE_USERNAME/KAGGLE_KEY tanımlı değil)" for s in KAGGLE_SETS}
    return {s: "kimlik bilgisi mevcut; indirme için `kaggle` CLI gerekir" for s in KAGGLE_SETS}


def build_fixture(source: Path, rows: int = FIXTURE_ROWS) -> Path:
    """Stratified (by target and SEX) sample of the Taiwan set for offline tests."""
    frame = pd.read_csv(source)
    strata = frame["default"].astype(str) + "_" + frame["SEX"].astype(str)
    fraction = rows / len(frame)
    sample = (
        frame.groupby(strata, group_keys=False)
        .apply(lambda g: g.sample(frac=fraction, random_state=FIXTURE_SEED))
        .sort_index()
    )
    header = (
        f"# Stratified sample (n={len(sample)}) of the UCI 'Default of Credit Card Clients' dataset.\n"
        "# Source: Yeh, I. (2009). UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C55S3H\n"
        "# License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
        "Column 'ID' dropped; target renamed to 'default'.\n"
    )
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(header + sample.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    return FIXTURE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="*", default=list(DATASETS))
    parser.add_argument("--fixture", action="store_true", help="rebuild the test fixture")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    manifest: dict[str, object] = {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "datasets": {name: fetch(name, args.force) for name in args.sets},
        "kaggle": kaggle_status(),
    }
    if args.fixture and "uci_taiwan" in args.sets:
        path = build_fixture(EXTERNAL / "uci_taiwan" / "data.csv")
        manifest["fixture_sha256"] = sha256(path.read_bytes())
    (EXTERNAL / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
