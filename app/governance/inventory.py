"""Model inventory, model cards, champion/challenger and rule-set governance.

* Inventory rows are synchronised from the artifact metadata at start-up.
* Model cards (purpose, data, features, metrics, limitations, fairness,
  approval status) render as Markdown and PDF.
* Champion/challenger: the challenger shadow-scores every decision (stored
  on the decision); promotion needs two different ``model_yoneticisi``
  approvals (four-eyes).
* Rule-set changes: a draft version is back-tested on historical decision
  snapshots (approval rate, expected loss) before two approvals activate it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import PROJECT_ROOT, get_settings
from app.core.rules import load_pricing, parse_policy
from app.db.audit import append_audit
from app.db.models import Decision, ModelRecord, RuleSet, utcnow
from app.decisioning.features import FEATURE_LABELS, MODEL_FEATURES
from app.decisioning.models import get_models

REQUIRED_APPROVALS = 2


class GovernanceError(ValueError):
    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def sync_inventory(session: Session) -> list[ModelRecord]:
    models = get_models()
    entries = [
        (models.pd_model.meta, "PD modeli (monotonik LightGBM, izotonik kalibrasyon)", "champion"),
        (models.scorecard.meta, "WoE skor kartı (optbinning + lojistik)", "champion"),
        (models.challenger.meta, "Challenger PD modeli (gölge skor)", "challenger"),
    ]
    rows = []
    for meta, name, role in entries:
        row = session.get(ModelRecord, meta["version"])
        if row is None:
            row = ModelRecord(
                id=meta["version"],
                name=name,
                kind=meta["kind"],
                version=meta["version"],
                role=role,
                metrics=meta.get("metrics", {}),
                artifact_path=str(get_settings().models_path),
            )
            session.add(row)
        row.metrics = meta.get("metrics", {})
        rows.append(row)
    fairness_path = get_settings().models_path / "fairness.json"
    if fairness_path.is_file():
        fairness = json.loads(fairness_path.read_text(encoding="utf-8"))
        pd_row = session.get(ModelRecord, models.pd_model.version) or rows[0]
        pd_row.card = {**(pd_row.card or {}), "fairness": fairness}
    session.flush()
    return rows


def model_card(row: ModelRecord) -> dict[str, Any]:
    models = get_models()
    meta = {m.version: m.meta for m in (models.pd_model, models.scorecard, models.challenger)}.get(
        row.id, {}
    )
    return {
        "model_id": row.id,
        "name": row.name,
        "role": row.role,
        "status": row.status,
        "purpose": "Bireysel ihtiyaç/taşıt kredisi başvurularında 12 ay içinde 90+ gün gecikme olasılığının (PD) tahmini.",
        "intended_use": "Hibrit karar motorunda politika kurallarıyla birlikte; bağlayıcı kararlar yetki matrisi ve insan denetimi altında.",
        "data": meta.get(
            "training", {"note": "Sentetik, seed'li popülasyon (gerçek kişi verisi yok)."}
        ),
        "features": [
            {
                "name": f,
                "label": FEATURE_LABELS.get(f, f),
                "monotone": (meta.get("monotone_constraints") or {}).get(f),
            }
            for f in meta.get("features", MODEL_FEATURES)
        ],
        "excluded_attributes": ["cinsiyet", "yaş bandı", "il (yalnızca izleme amaçlı)"],
        "metrics": row.metrics,
        "fairness": (row.card or {}).get("fairness"),
        "limitations": [
            "Sentetik veri gerçek portföylerden daha ayrıştırıcıdır; canlıya almadan önce gerçek veride yeniden eğitim ve doğrulama gerekir.",
            "Makro-ekonomik şoklar ve politika değişiklikleri eğitim dağılımı dışında kalabilir (PSI izlemesi zorunlu).",
            "İnce dosyalı başvurularda açık bankacılık verisi yoksa belirsizlik artar.",
        ],
        "monitoring": "Aylık PSI/CSI drift, AIR (4/5 kuralı) ve champion/challenger karşılaştırması.",
        "approvals": row.approvals,
        "regulatory": [
            "BDDK Kredi İşlemleri Yönetmeliği",
            "KVKK m.11",
            "EU AI Act Annex III 5(b) (yüksek risk)",
            "SR 26-2 model risk yönetimi beklentisi",
        ],
    }


def card_markdown(card: dict[str, Any]) -> str:
    lines = [
        f"# Model Kartı — {card['name']} ({card['model_id']})",
        "",
        f"**Rol:** {card['role']} · **Durum:** {card['status']}",
        "",
        "## Amaç",
        card["purpose"],
        "",
        "## Kullanım",
        card["intended_use"],
        "",
        "## Metrikler",
    ]
    for key in ("auc", "gini", "ks", "brier", "default_rate", "n"):
        if key in card["metrics"]:
            lines.append(f"- {key.upper()}: {card['metrics'][key]}")
    lines += ["", "## Değişkenler"] + [
        f"- {f['label']} (`{f['name']}`), monotonluk: {f['monotone']}" for f in card["features"]
    ]
    lines += ["", "## Modelde kullanılmayan korunan özellikler"] + [
        f"- {a}" for a in card["excluded_attributes"]
    ]
    if card.get("fairness"):
        lines += ["", "## Adillik"] + [
            f"- {attr}: asgari AIR {s['min_air']}"
            for attr, s in card["fairness"].get("attributes", {}).items()
        ]
    lines += ["", "## Sınırlamalar"] + [f"- {x}" for x in card["limitations"]]
    lines += ["", "## İzleme", card["monitoring"], "", "## Düzenleyici çerçeve"] + [
        f"- {x}" for x in card["regulatory"]
    ]
    return "\n".join(lines)


def card_pdf(card: dict[str, Any], path: Path) -> Path:
    from xml.sax.saxutils import escape

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    from app.core.pdf import FONT_BOLD, FONT_REGULAR, register_fonts

    register_fonts()
    base = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=base["Heading2"], fontName=FONT_BOLD)
    b = ParagraphStyle("b", parent=base["BodyText"], fontName=FONT_REGULAR)
    story = []
    for line in card_markdown(card).splitlines():
        if line.startswith("#"):
            story.append(Paragraph(escape(line.lstrip("# ")), h))
        elif line.strip():
            story.append(Paragraph(escape(line.replace("**", "")), b))
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return path


def champion_challenger(session: Session, limit: int = 2000) -> dict[str, Any]:
    rows = session.execute(
        select(Decision.pd, Decision.challenger)
        .where(Decision.kind == "engine")
        .order_by(Decision.created_at.desc())
        .limit(limit)
    ).all()
    pairs = [
        (pd, ch.get("pd")) for pd, ch in rows if pd is not None and ch and ch.get("pd") is not None
    ]
    if not pairs:
        return {"decisions": 0}
    champ = [p for p, _ in pairs]
    chall = [c for _, c in pairs]
    cutoff = 0.05
    agree = sum(1 for p, c in pairs if (p <= cutoff) == (c <= cutoff))
    return {
        "decisions": len(pairs),
        "mean_pd": {
            "champion": round(sum(champ) / len(champ), 4),
            "challenger": round(sum(chall) / len(chall), 4),
        },
        "approval_rate_at_cutoff": {
            "champion": round(sum(p <= cutoff for p in champ) / len(champ), 4),
            "challenger": round(sum(c <= cutoff for c in chall) / len(chall), 4),
        },
        "decision_agreement": round(agree / len(pairs), 4),
        "offline_metrics": {
            m.version: m.meta.get("metrics", {})
            for m in (get_models().pd_model, get_models().challenger)
        },
    }


def approve_promotion(session: Session, model_id: str, approver: str) -> ModelRecord:
    row = session.get(ModelRecord, model_id)
    if row is None:
        raise GovernanceError("model bulunamadı", 404)
    if row.role != "challenger":
        raise GovernanceError("yalnızca challenger modeller terfi ettirilebilir")
    approvals = list(row.approvals or [])
    if approver in {a["by"] for a in approvals}:
        raise GovernanceError("dört göz: aynı kişi ikinci kez onay veremez", 403)
    approvals.append({"by": approver, "at": utcnow().isoformat()})
    row.approvals = approvals
    append_audit(
        session,
        actor=approver,
        action="MODEL_PROMOTION_APPROVAL",
        entity_type="model",
        entity_id=model_id,
        payload={"approvals": len(approvals)},
    )
    if len(approvals) >= REQUIRED_APPROVALS:
        for other in session.execute(
            select(ModelRecord).where(ModelRecord.role == "champion", ModelRecord.kind == row.kind)
        ).scalars():
            other.role = "retired"
        row.role = "champion"
        row.status = "TERFI_ONAYLANDI"
        append_audit(
            session,
            actor=approver,
            action="MODEL_PROMOTED",
            entity_type="model",
            entity_id=model_id,
        )
    return row


# ------------------------------------------------------------------ rule sets
def backtest_rule_set(session: Session, content: str, limit: int = 2000) -> dict[str, Any]:
    from app.decisioning.engine import decide

    candidate = parse_policy(content)
    snapshots = [
        (s, o)
        for s, o in session.execute(
            select(Decision.feature_snapshot, Decision.outcome)
            .where(Decision.kind == "engine")
            .order_by(Decision.created_at.desc())
            .limit(limit)
        ).all()
        if s
    ]
    if not snapshots:
        return {"decisions": 0, "note": "geriye dönük test için karar geçmişi yok"}
    models, pricing = get_models(), load_pricing()
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    el_before = el_after = 0.0
    changed = 0
    for snapshot, outcome in snapshots:
        result = decide(snapshot, policy=candidate, models=models, pricing_cfg=pricing)
        before[outcome] = before.get(outcome, 0) + 1
        after[result.outcome] = after.get(result.outcome, 0) + 1
        amount = float(snapshot.get("requested_amount", 0.0))
        lgd = pricing.product(snapshot.get("product", "IHTIYAC")).lgd
        if outcome == "OTOMATIK_ONAY":
            el_before += result.pd_requested * lgd * amount
        if result.outcome == "OTOMATIK_ONAY":
            el_after += result.pd * lgd * result.offer_amount
        changed += outcome != result.outcome
    n = len(snapshots)
    return {
        "decisions": n,
        "version": candidate.version,
        "outcomes_before": before,
        "outcomes_after": after,
        "approval_rate_before": round(before.get("OTOMATIK_ONAY", 0) / n, 4),
        "approval_rate_after": round(after.get("OTOMATIK_ONAY", 0) / n, 4),
        "expected_loss_before": round(el_before, 2),
        "expected_loss_after": round(el_after, 2),
        "changed_decisions": changed,
    }


def submit_rule_set(session: Session, version: str, content: str, actor: str) -> RuleSet:
    policy = parse_policy(content)
    if policy.version != version:
        raise GovernanceError("YAML içindeki version alanı ile istek uyuşmuyor", 422)
    if session.get(RuleSet, version) is not None:
        raise GovernanceError("bu versiyon zaten mevcut")
    row = RuleSet(
        id=version, content=content, status="TASLAK", backtest=backtest_rule_set(session, content)
    )
    session.add(row)
    append_audit(
        session,
        actor=actor,
        action="RULESET_SUBMITTED",
        entity_type="rule_set",
        entity_id=version,
        payload={"backtest": row.backtest},
    )
    return row


def approve_rule_set(session: Session, version: str, approver: str) -> RuleSet:
    row = session.get(RuleSet, version)
    if row is None:
        raise GovernanceError("kural seti bulunamadı", 404)
    if row.status != "TASLAK":
        raise GovernanceError("kural seti taslak durumda değil")
    approvals = list(row.approvals or [])
    if approver in {a["by"] for a in approvals}:
        raise GovernanceError("dört göz: aynı kişi ikinci kez onay veremez", 403)
    approvals.append({"by": approver, "at": utcnow().isoformat()})
    row.approvals = approvals
    if len(approvals) >= REQUIRED_APPROVALS:
        for active in session.execute(
            select(RuleSet).where(RuleSet.status == "YURURLUKTE")
        ).scalars():
            active.status = "ARSIV"
        row.status = "YURURLUKTE"
        row.activated_at = utcnow()
        append_audit(
            session,
            actor=approver,
            action="RULESET_ACTIVATED",
            entity_type="rule_set",
            entity_id=version,
        )
    return row


def default_policy_path() -> Path:
    return PROJECT_ROOT / "rules" / "policy_v1.yaml"
