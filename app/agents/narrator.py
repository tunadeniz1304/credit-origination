"""Credit narratives: prompt chain + deterministic Turkish templates.

The narrator explains a decision that the deterministic engine already made;
it never decides. The live chain is:

1. ``analysis``          — structured JSON (strengths/risks as cited claims)
2. ``committee_summary`` — receives step 1's JSON; every number cited ``[f:id]``
3. ``applicant_letter``  — plain Turkish for the applicant (reason codes, KVKK
   m.11 objection right when the outcome is adverse)

plus ``missing_documents_letter``. In demo mode (or on fallback) the same
outputs are rendered from templates over the identical context, so a demo
report reads like a real one and never contains placeholder text.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.citations import (
    strip_citations,
    validate_cited_text,
    validate_claims,
    validate_plain_text,
)
from app.agents.llm_service import Generation, LLMService
from app.agents.redaction import Redactor

Outcome = Literal[
    "OTOMATIK_ONAY", "OTOMATIK_RET", "UZMAN_INCELEMESI", "ONAYLANDI", "REDDEDILDI", "BELGE_EKSIK"
]

OUTCOME_LABELS: dict[str, str] = {
    "OTOMATIK_ONAY": "Otomatik Onay",
    "OTOMATIK_RET": "Otomatik Ret",
    "UZMAN_INCELEMESI": "Uzman İncelemesi",
    "ONAYLANDI": "Onay (Uzman Kararı)",
    "REDDEDILDI": "Ret (Uzman Kararı)",
    "BELGE_EKSIK": "Belge Eksik",
}


# ----------------------------------------------------------------- formatting
def fmt_number(value: float, decimals: int = 2) -> str:
    """Turkish number format: ``1234567.5`` -> ``1.234.567,50``."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_tl(value: float) -> str:
    return f"{fmt_number(value, 2)} TL"


def fmt_pct(ratio: float, decimals: int = 1) -> str:
    return f"%{fmt_number(ratio * 100, decimals)}"


class Fact(BaseModel):
    """One citable field of the decision context."""

    id: str
    label: str
    value: float | int | str
    unit: Literal["TL", "%", "ay", "puan", "gün", "adet", "yıl", ""] = ""

    @property
    def display(self) -> str:
        if isinstance(self.value, str):
            return self.value
        if self.unit == "TL":
            return fmt_tl(float(self.value))
        if self.unit == "%":
            return fmt_pct(float(self.value))
        if self.unit in ("ay", "gün", "adet", "yıl"):
            return f"{fmt_number(float(self.value), 0)} {self.unit}"
        if self.unit == "puan":
            return f"{fmt_number(float(self.value), 0)} puan"
        return fmt_number(float(self.value), 2)

    @property
    def cite(self) -> str:
        return f"{self.display} [{self.id}]"


class ReasonText(BaseModel):
    code: str
    text: str


class NarrativeContext(BaseModel):
    """Everything a narrative may mention; built by the decision engine."""

    application_id: str
    applicant_name: str
    outcome: Outcome
    conditional: bool = False
    facts: list[Fact] = Field(default_factory=list)
    reason_codes: list[ReasonText] = Field(default_factory=list)
    counterfactuals: list[str] = Field(default_factory=list)
    missing_documents: list[str] = Field(default_factory=list)
    fraud_flags: list[str] = Field(default_factory=list)
    objection_deadline_days: int = 30
    review_sla_hours: int = 24
    redactor_fields: dict[str, str] = Field(default_factory=dict)

    def fact_map(self) -> dict[str, float | int | str]:
        return {f.id: f.value for f in self.facts}

    def get(self, fid: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fid), None)

    def redactor(self) -> Redactor:
        fields = self.redactor_fields
        return Redactor.for_applicant(
            name=self.applicant_name,
            identity_no=fields.get("identity_no"),
            phone=fields.get("phone"),
            iban=fields.get("iban"),
            address=fields.get("address"),
            email=fields.get("email"),
        )

    @property
    def adverse(self) -> bool:
        return self.outcome in ("OTOMATIK_RET", "REDDEDILDI")


class Claim(BaseModel):
    text: str
    field_ids: list[str] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    """Step 1 structured output."""

    summary: str
    strengths: list[Claim] = Field(default_factory=list)
    risks: list[Claim] = Field(default_factory=list)
    recommendation: str = ""


class NarrativeBundle(BaseModel):
    analysis: AnalysisOutput
    committee_summary: str
    applicant_letter: str
    modes: dict[str, str] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)

    @property
    def overall_mode(self) -> str:
        values = set(self.modes.values())
        if "fallback" in values:
            return "fallback"
        return "live" if values == {"live"} else "demo"


_SYSTEM_ANALYST = (
    "Sen Türkiye'de faaliyet gösteren bir bankanın kredi tahsis analistisin. BDDK Kredi "
    "İşlemleri Yönetmeliği ve banka iç kredi politikası çerçevesinde nesnel, resmi ve "
    "gerekçeli Türkçe yazarsın. Kredi kararını SEN VERMEZSİN: onay/ret/limit/faiz "
    "deterministik karar motorundan gelir, sen yalnızca açıklarsın. Bağlamda olmayan "
    "hiçbir sayı veya iddia yazma."
)


# --------------------------------------------------------------- demo renderers
def _pick(ctx: NarrativeContext, *ids: str) -> list[Fact]:
    return [fact for fid in ids if (fact := ctx.get(fid)) is not None]


def render_analysis(ctx: NarrativeContext) -> AnalysisOutput:
    strengths: list[Claim] = []
    risks: list[Claim] = []
    score = ctx.get("f:bureau.score")
    if score is not None and isinstance(score.value, int | float):
        target = strengths if float(score.value) >= 1100 else risks
        target.append(
            Claim(
                text=f"KKB kredi notu {score.display} seviyesindedir.",
                field_ids=[score.id],
            )
        )
    dsr, limit = ctx.get("f:dsr"), ctx.get("f:policy.max_dsr")
    if dsr is not None and limit is not None:
        within = float(dsr.value) <= float(limit.value)
        text = f"Borç servis oranı {dsr.display} olup politika sınırı {limit.display} " + (
            "içinde kalmaktadır." if within else "aşılmaktadır."
        )
        (strengths if within else risks).append(Claim(text=text, field_ids=[dsr.id, limit.id]))
    pd = ctx.get("f:model.pd")
    if pd is not None:
        low = float(pd.value) <= 0.05
        (strengths if low else risks).append(
            Claim(
                text=f"Model bazlı 12 aylık temerrüt olasılığı {pd.display} olarak hesaplanmıştır.",
                field_ids=[pd.id],
            )
        )
    tenure = ctx.get("f:employment.months")
    if tenure is not None:
        stable = float(tenure.value) >= 24
        (strengths if stable else risks).append(
            Claim(
                text=f"Mevcut işyerindeki çalışma süresi {tenure.display} düzeyindedir.",
                field_ids=[tenure.id],
            )
        )
    income_cv = ctx.get("f:cashflow.income_cv")
    if income_cv is not None:
        steady = float(income_cv.value) <= 0.25
        (strengths if steady else risks).append(
            Claim(
                text=f"Hesap hareketlerinde gelir değişkenlik katsayısı {income_cv.display}.",
                field_ids=[income_cv.id],
            )
        )
    for flag in ctx.fraud_flags:
        risks.append(Claim(text=f"Belge/kimlik kontrol bulgusu: {flag}.", field_ids=[]))
    amount, term = ctx.get("f:loan.amount"), ctx.get("f:loan.term")
    summary_parts = [f"{ctx.application_id} numaralı başvuru"]
    if amount is not None and term is not None:
        summary_parts.append(f"{amount.display} tutarında ve {term.display} vadeli")
    summary = " ".join(summary_parts) + " bireysel kredi talebine ilişkindir."
    recommendation = {
        "OTOMATIK_ONAY": "Karar motoru sonucu onay yönündedir; teklif koşullarıyla kullandırım "
        "önerilir.",
        "OTOMATIK_RET": "Karar motoru sonucu ret yönündedir; başvurana gerekçe kodları ve "
        "itiraz hakkı bildirilmelidir.",
        "UZMAN_INCELEMESI": "Başvuru gri bölgededir; yetki matrisine göre uzman incelemesi "
        "gereklidir.",
    }.get(ctx.outcome, "Uzman kararı doğrultusunda işlem yapılmalıdır.")
    return AnalysisOutput(
        summary=summary, strengths=strengths, risks=risks, recommendation=recommendation
    )


def render_committee_summary(ctx: NarrativeContext, analysis: AnalysisOutput) -> str:
    lines = [
        f"KREDİ KOMİTESİ ÖZETİ — {ctx.application_id}",
        f"Karar motoru sonucu: {OUTCOME_LABELS.get(ctx.outcome, ctx.outcome)}"
        + (" (koşullu)" if ctx.conditional else ""),
        "",
    ]
    income, amount, term = (
        _pick(ctx, "f:income.monthly"),
        ctx.get("f:loan.amount"),
        ctx.get("f:loan.term"),
    )
    if income and amount and term:
        lines.append(
            f"Başvuranın doğrulanan aylık net geliri {income[0].cite}; talep edilen kredi "
            f"{amount.cite}, vade {term.cite}."
        )
    for fid, template in (
        ("f:dsr", "Yeni taksit dahil borç servis oranı {v}"),
        ("f:bureau.score", "KKB kredi notu {v}"),
        ("f:model.pd", "Kalibre edilmiş 12 aylık temerrüt olasılığı (PD) {v}"),
        ("f:scorecard.points", "Skor kartı puanı {v}"),
        ("f:limit.max_amount", "Onaylanabilir azami tutar {v}"),
        ("f:pricing.annual_rate", "Risk bazlı yıllık akdi faiz {v}"),
        ("f:offer.instalment", "Aylık taksit {v}"),
        ("f:pricing.apr", "Yıllık maliyet oranı {v}"),
    ):
        fact = ctx.get(fid)
        if fact is not None:
            lines.append(template.format(v=fact.cite) + ".")
    if analysis.strengths:
        lines += ["", "Güçlü yönler:"]
        lines += [
            f"- {c.text} " + " ".join(f"[{fid}]" for fid in c.field_ids) for c in analysis.strengths
        ]
    if analysis.risks:
        lines += ["", "Risk unsurları:"]
        lines += [
            f"- {c.text} " + " ".join(f"[{fid}]" for fid in c.field_ids) for c in analysis.risks
        ]
    if ctx.reason_codes:
        lines += ["", "Gerekçe kodları: " + ", ".join(rc.code for rc in ctx.reason_codes)]
    lines += ["", f"Öneri: {analysis.recommendation}"]
    return "\n".join(line.rstrip() for line in lines)


def render_applicant_letter(ctx: NarrativeContext) -> str:
    greeting = f"Sayın {ctx.applicant_name},"
    closing = ["", "Saygılarımızla,", "Bireysel Krediler Tahsis Birimi"]
    amount, term = ctx.get("f:loan.amount"), ctx.get("f:loan.term")
    request = (
        f"{amount.display} tutarlı ve {term.display} vadeli" if amount and term else "bireysel"
    )
    if ctx.outcome in ("OTOMATIK_ONAY", "ONAYLANDI"):
        body = [
            greeting,
            "",
            f"{ctx.application_id} numaralı {request} kredi başvurunuz olumlu değerlendirilmiştir.",
        ]
        offer_amount = ctx.get("f:offer.amount")
        if ctx.conditional and offer_amount is not None:
            body.append(
                f"Talep ettiğiniz tutar yerine size {offer_amount.display} tutarında bir teklif "
                "sunulmaktadır; bu tutar ödeme gücünüz ve kredi politikamız dikkate alınarak "
                "belirlenmiştir."
            )
        details = [
            (fid, label)
            for fid, label in (
                ("f:offer.amount", "Kredi tutarı"),
                ("f:offer.term", "Vade"),
                ("f:offer.instalment", "Aylık taksit"),
                ("f:pricing.annual_rate", "Yıllık akdi faiz oranı"),
                ("f:pricing.apr", "Yıllık maliyet oranı"),
            )
            if ctx.get(fid) is not None
        ]
        if details:
            body += ["", "Teklif koşulları:"]
            body += [f"- {label}: {ctx.get(fid).display}" for fid, label in details]  # type: ignore[union-attr]
        body += [
            "",
            "Teklifinizi başvuru portalı üzerinden inceleyip geçerlilik süresi içinde kabul "
            "edebilirsiniz. Sözleşme öncesi bilgi formu kabulünüzün ardından iletilecektir.",
        ]
        return "\n".join(body + closing)
    if ctx.outcome == "UZMAN_INCELEMESI":
        body = [
            greeting,
            "",
            f"{ctx.application_id} numaralı {request} kredi başvurunuz, ek değerlendirme amacıyla "
            "kredi uzmanlarımızın incelemesine alınmıştır. Bu aşamada kararınız otomatik olarak "
            "verilmemiştir; başvurunuz bir uzman tarafından değerlendirilecektir.",
            "",
            f"Değerlendirmenin en geç {ctx.review_sla_hours} saat içinde tamamlanması "
            "hedeflenmektedir. Ek belge ihtiyacı olması halinde sizinle iletişime geçilecektir.",
        ]
        return "\n".join(body + closing)
    if ctx.outcome == "BELGE_EKSIK":
        return render_missing_documents_letter(ctx)
    body = [
        greeting,
        "",
        f"{ctx.application_id} numaralı {request} kredi başvurunuz değerlendirilmiş, ancak bu "
        "aşamada olumlu sonuçlandırılamamıştır.",
    ]
    if ctx.reason_codes:
        body += ["", "Kararın başlıca gerekçeleri:"]
        body += [f"- {rc.text}" for rc in ctx.reason_codes]
    if ctx.counterfactuals:
        body += ["", "Aşağıdaki değişikliklerle yeniden başvurmanız halinde sonuç olumlu olabilir:"]
        body += [f"- {cf}" for cf in ctx.counterfactuals]
    body += [
        "",
        "6698 sayılı Kişisel Verilerin Korunması Kanunu'nun 11. maddesi uyarınca, işlenen "
        "verilerinizin münhasıran otomatik sistemler vasıtasıyla analiz edilmesi suretiyle "
        "aleyhinize bir sonucun ortaya çıkmasına itiraz etme hakkına sahipsiniz. İtirazınızı "
        f"bu bildirimden itibaren {ctx.objection_deadline_days} gün içinde başvuru portalındaki "
        "'İtiraz Et' adımıyla iletebilirsiniz; itirazınız bir kredi uzmanı tarafından insan "
        "incelemesiyle değerlendirilir ve sonucu size yazılı olarak bildirilir.",
    ]
    return "\n".join(body + closing)


def render_missing_documents_letter(ctx: NarrativeContext) -> str:
    lines = [
        f"Sayın {ctx.applicant_name},",
        "",
        f"{ctx.application_id} numaralı kredi başvurunuzun değerlendirmesine devam edebilmemiz "
        "için aşağıdaki belgelerin tarafımıza iletilmesi gerekmektedir:",
        "",
    ]
    lines += [f"- {doc}" for doc in ctx.missing_documents]
    lines += [
        "",
        "Belgelerinizi başvuru portalındaki 'Belgelerim' bölümünden PDF veya görüntü olarak "
        "yükleyebilirsiniz. Tüm belgeler tamamlandığında başvurunuz otomatik olarak "
        "değerlendirmeye alınacaktır. Belgelerin 10 iş günü içinde iletilmemesi halinde "
        "başvurunuz iptal edilebilir.",
        "",
        "Saygılarımızla,",
        "Kredi Operasyon Birimi",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- narrator
def _context_block(ctx: NarrativeContext) -> str:
    payload: dict[str, Any] = {
        "basvuru_no": ctx.application_id,
        "karar_motoru_sonucu": ctx.outcome,
        "kosullu": ctx.conditional,
        "alanlar": [
            {"field_id": f.id, "etiket": f.label, "deger": f.value, "birim": f.unit}
            for f in ctx.facts
        ],
        "gerekce_kodlari": [rc.model_dump() for rc in ctx.reason_codes],
        "karsi_olgusal_oneriler": ctx.counterfactuals,
        "sahtecilik_bulgulari": ctx.fraud_flags,
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


class Narrator:
    """Runs the three-step chain (or its deterministic twin)."""

    def __init__(self, service: LLMService | None = None) -> None:
        self.service = service or LLMService()

    async def analysis(self, ctx: NarrativeContext) -> Generation[AnalysisOutput]:
        facts = ctx.fact_map()

        def guard(parsed: AnalysisOutput) -> list[str]:
            claims = [(c.text, c.field_ids) for c in parsed.strengths + parsed.risks]
            result = validate_claims(claims, facts)
            if not claims:
                return ["no claims produced"]
            return result.errors

        messages = [
            {"role": "system", "content": _SYSTEM_ANALYST},
            {
                "role": "user",
                "content": (
                    "Aşağıdaki karar bağlamını analiz et ve YALNIZCA şu JSON şemasında yanıt ver: "
                    '{"summary": str, "strengths": [{"text": str, "field_ids": [str]}], '
                    '"risks": [{"text": str, "field_ids": [str]}], "recommendation": str}. '
                    "Her iddiadaki her sayı, field_ids içinde atıf yaptığın alanın değeriyle "
                    "aynı olmalı.\n\n" + _context_block(ctx)
                ),
            },
        ]
        return await self.service.generate(
            "analysis",
            messages,
            fallback=lambda: render_analysis(ctx).model_dump_json(),
            schema=AnalysisOutput,
            guard=guard,
            redactor=ctx.redactor(),
            application_id=ctx.application_id,
        )

    async def committee_summary(
        self, ctx: NarrativeContext, analysis: AnalysisOutput
    ) -> Generation[Any]:
        facts = ctx.fact_map()
        messages = [
            {"role": "system", "content": _SYSTEM_ANALYST},
            {
                "role": "user",
                "content": (
                    "Önceki adımda üretilen yapılandırılmış analiz aşağıdadır. Bunu kullanarak "
                    "kredi komitesi için resmi, kısa (en fazla 12 satır) bir özet yaz. Her sayının "
                    "hemen arkasına kaynağını [f:alan_id] biçiminde ekle.\n\nANALİZ:\n"
                    + analysis.model_dump_json(indent=1)
                    + "\n\nBAĞLAM:\n"
                    + _context_block(ctx)
                ),
            },
        ]
        return await self.service.generate(
            "committee_summary",
            messages,
            fallback=lambda: render_committee_summary(ctx, analysis),
            guard=lambda text: validate_cited_text(text, facts).errors,
            redactor=ctx.redactor(),
            application_id=ctx.application_id,
        )

    async def applicant_letter(self, ctx: NarrativeContext) -> Generation[Any]:
        from app.agents.citations import extract_numbers

        facts: dict[str, Any] = dict(ctx.fact_map())
        # Numbers quoted by reason-code / counterfactual texts are part of the context too.
        quoted = [rc.text for rc in ctx.reason_codes] + ctx.counterfactuals
        for i, (_, readings) in enumerate(n for text in quoted for n in extract_numbers(text)):
            for j, value in enumerate(sorted(readings)):
                facts[f"f:quoted.{i}.{j}"] = value
        adverse_note = (
            " Sonuç olumsuz olduğu için gerekçe kodlarını sade dille açıkla, karşı-olgusal "
            "önerileri aktar ve KVKK m.11 kapsamındaki itiraz hakkını ve insan incelemesi "
            "talep etme yolunu mutlaka belirt."
            if ctx.adverse
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": "Sen bir bankanın müşteri iletişimi uzmanısın. Sade, saygılı ve "
                "anlaşılır Türkçe yazarsın; teknik terim ve alan kodu kullanmazsın.",
            },
            {
                "role": "user",
                "content": (
                    "Başvurana hitaben bir karar bildirim mektubu yaz. Mektupta bağlamda olmayan "
                    "sayı kullanma." + adverse_note + "\n\nBAĞLAM:\n" + _context_block(ctx)
                ),
            },
        ]
        generation: Any = await self.service.generate(
            "applicant_letter",
            messages,
            fallback=lambda: render_applicant_letter(ctx),
            guard=lambda text: (
                validate_plain_text(text, facts).errors
                + (
                    ["missing KVKK m.11 objection notice"]
                    if ctx.adverse and "itiraz" not in text.lower()
                    else []
                )
            ),
            redactor=ctx.redactor(),
            application_id=ctx.application_id,
        )
        generation.text = strip_citations(generation.text)
        return generation

    async def missing_documents_letter(self, ctx: NarrativeContext) -> Generation[Any]:
        messages = [
            {
                "role": "system",
                "content": "Sen bir bankanın kredi operasyon uzmanısın; sade ve resmi Türkçe yaz.",
            },
            {
                "role": "user",
                "content": (
                    "Başvurana eksik belgeleri listeleyen kısa bir yazı hazırla. Eksik belgeler: "
                    + "; ".join(ctx.missing_documents)
                    + f". Başvuru no: {ctx.application_id}."
                ),
            },
        ]
        return await self.service.generate(
            "missing_documents_letter",
            messages,
            fallback=lambda: render_missing_documents_letter(ctx),
            guard=lambda text: (
                []
                if all(doc.split(" ")[0].lower() in text.lower() for doc in ctx.missing_documents)
                else ["missing document list incomplete"]
            ),
            redactor=ctx.redactor(),
            application_id=ctx.application_id,
        )

    async def run_chain(self, ctx: NarrativeContext) -> NarrativeBundle:
        """Step 1 → step 2 (receives step 1 output) → step 3."""
        analysis = await self.analysis(ctx)
        parsed = analysis.parsed or render_analysis(ctx)
        summary = await self.committee_summary(ctx, parsed)
        letter = await self.applicant_letter(ctx)
        modes = {
            "analysis": analysis.mode,
            "committee_summary": summary.mode,
            "applicant_letter": letter.mode,
        }
        errors = {
            name: gen.error_kind
            for name, gen in (
                ("analysis", analysis),
                ("committee_summary", summary),
                ("applicant_letter", letter),
            )
            if gen.error_kind
        }
        return NarrativeBundle(
            analysis=parsed,
            committee_summary=summary.text,
            applicant_letter=letter.text,
            modes=modes,
            errors={k: str(v) for k, v in errors.items()},
        )
