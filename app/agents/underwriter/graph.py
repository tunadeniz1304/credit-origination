"""Supervised underwriter agent (nCino Banking Advisor style) on LangGraph.

Graph: ``plan → act → plan … → draft → END`` with at most eight tool steps.

* **plan** asks the LLM for the next tool via OpenAI tool calling; if the
  endpoint rejects ``tools`` the agent switches to a ReAct text protocol
  (``Eylem: <araç>`` / ``Girdi: {...}``). In demo mode a deterministic plan
  visits every tool once.
* **act** executes the read-only tool and writes an audit entry per step.
* **draft** writes the *Kredi Tahsis Memorandumu*: every number must cite a
  ``field_id`` collected by the tools (citation guard → one repair → template
  fallback).

The agent produces a **recommendation only**; the binding decision stays with
the engine and the authorised specialist / committee.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.citations import validate_cited_text
from app.agents.llm import LLMCallError, LLMResponse, ToolsNotSupportedError
from app.agents.llm_service import LLMService
from app.agents.narrator import Fact
from app.agents.redaction import Redactor
from app.agents.underwriter.tools import TOOL_SPECS, UnderwriterTools, openai_tools
from app.db.audit import append_audit
from app.db.models import Application

MAX_STEPS = 8
DEMO_PLAN: list[tuple[str, dict[str, Any]]] = [
    ("get_application", {}),
    ("get_documents_fields", {}),
    ("get_bureau_report", {}),
    ("get_cashflow_features", {}),
    ("run_decision", {}),
    ("get_reason_codes", {}),
    ("get_pricing", {}),
    ("search_policy", {"query": "borç servis oranı yetki matrisi dört göz"}),
]
_SYSTEM = (
    "Sen bir bankada kıdemli kredi analistisin. Araçları kullanarak başvuruyu incele ve ardından "
    "draft_memo aracını çağır. Kredi kararı vermezsin; yalnızca öneride bulunursun. En fazla 8 araç çağrısı yap."
)
_REACT_HINT = (
    "Araç çağırma desteklenmiyor. Her adımda yalnızca şu biçimde yanıt ver:\n"
    "Eylem: <araç_adı>\nGirdi: <JSON>\nYeterli bilgi toplandığında 'Eylem: draft_memo' yaz. Araçlar: "
)


class AgentStep(BaseModel):
    step: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    summary: str


class MemoSection(BaseModel):
    title: str
    text: str


class CreditMemo(BaseModel):
    application_id: str
    sections: list[MemoSection]
    recommendation: str
    citations: list[str]
    steps: list[AgentStep]
    mode: str
    protocol: str
    error_kind: str | None = None


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    steps: list[AgentStep]
    pending: tuple[str, dict[str, Any]] | None
    done: bool
    protocol: str


def _parse_react(text: str) -> tuple[str, dict[str, Any]] | None:
    match = re.search(r"(?:Eylem|Action)\s*:\s*([a-z_]+)", text)
    if not match:
        return None
    args: dict[str, Any] = {}
    raw = re.search(r"(?:Girdi|Action Input)\s*:\s*(\{.*\})", text, re.DOTALL)
    if raw:
        try:
            args = json.loads(raw.group(1))
        except ValueError:
            args = {}
    return match.group(1), args


def _summarise(result: dict[str, Any]) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text[:160] + ("…" if len(text) > 160 else "")


# ------------------------------------------------------------------ memo template
def render_memo_sections(
    app: Application, facts: dict[str, Fact], obs: dict[str, dict[str, Any]]
) -> tuple[list[MemoSection], str]:
    def cite(fid: str) -> str:
        fact = facts.get(fid)
        return fact.cite if fact else "—"

    decision = obs.get("run_decision", {})
    outcome = decision.get("outcome", "—")
    sections = [
        MemoSection(
            title="Başvuran Profili ve Talep",
            text=f"{app.id} numaralı başvuruda {cite('f:loan.amount')} tutarında, {cite('f:loan.term')} vadeli "
            f"{app.product.lower()} kredisi talep edilmektedir. Beyan edilen aylık net gelir {cite('f:income.declared')}.",
        ),
        MemoSection(
            title="Gelir ve İstihdam Doğrulaması",
            text=f"SGK kaydına göre mevcut işyerindeki çalışma süresi {cite('f:employment.months')}. "
            + (
                f"Hesap hareketlerindeki ortalama aylık maaş girişi {cite('f:cashflow.avg_income')}."
                if "f:cashflow.avg_income" in facts
                else "Açık bankacılık verisi bulunmamaktadır."
            ),
        ),
        MemoSection(
            title="Borçluluk",
            text=(
                f"KKB kredi notu {cite('f:bureau.score')}. "
                if "f:bureau.score" in facts
                else "KKB kaydı bulunmamaktadır (ince dosya). "
            )
            + f"Aktif kredi sayısı {cite('f:bureau.active_loans')}, mevcut aylık taksitler {cite('f:bureau.instalments')}; "
            f"yeni taksit dahil borç servis oranı {cite('f:dsr')} ve politika sınırı {cite('f:policy.max_dsr')}.",
        ),
        MemoSection(
            title="Nakit Akışı",
            text=(
                f"Gelir değişkenliği {cite('f:cashflow.income_cv')}, son 12 ayda eksi bakiye {cite('f:cashflow.negative_days')}, "
                f"tasarruf oranı {cite('f:cashflow.savings_rate')}."
                if "f:cashflow.income_cv" in facts
                else "Nakit akışı analizi yapılamamıştır."
            ),
        ),
        MemoSection(
            title="Belge Bulguları",
            text=f"{cite('f:docs.count')} belge incelenmiştir; en yüksek sahtecilik skoru {cite('f:docs.max_fraud_score')}, "
            f"en düşük alan güveni {cite('f:docs.min_confidence')}.",
        ),
        MemoSection(
            title="Model ve Kural Sonuçları",
            text=f"Karar motoru sonucu {outcome}. Kalibre edilmiş 12 aylık temerrüt olasılığı {cite('f:model.pd')}, "
            f"skor kartı puanı {cite('f:scorecard.points')}. Tetiklenen kurallar: {', '.join(decision.get('fired_rules') or ['yok'])}.",
        ),
    ]
    reasons = obs.get("get_reason_codes", {}).get("reason_codes", [])
    risk_text = (
        "; ".join(r["code"] for r in reasons) if reasons else "belirgin olumsuz gerekçe kodu yok"
    )
    counter = obs.get("get_reason_codes", {}).get("counterfactuals", [])
    sections.append(
        MemoSection(
            title="Riskler ve Azaltıcılar",
            text=f"Gerekçe kodları: {risk_text}. "
            + (
                f"Azaltıcı öneri: {counter[0]}" if counter else "Ek azaltıcı öneri bulunmamaktadır."
            ),
        )
    )
    if "f:pricing.annual_rate" in facts:
        sections.append(
            MemoSection(
                title="Fiyatlama",
                text=f"Risk bazlı yıllık akdi faiz {cite('f:pricing.annual_rate')}, aylık taksit {cite('f:offer.instalment')}, "
                f"yıllık maliyet oranı {cite('f:pricing.apr')}, RAROC {cite('f:pricing.raroc')}.",
            )
        )
    policy_hits = obs.get("search_policy", {}).get("results", [])
    if policy_hits:
        sections.append(
            MemoSection(
                title="Politika Referansları",
                text="İlgili politika bölümleri: "
                + "; ".join(f"{h['title']} [kaynak:{h['chunk_id']}]" for h in policy_hits[:2])
                + ".",
            )
        )
    recommendation = {
        "OTOMATIK_ONAY": "Öneri: teklif koşullarıyla kullandırım uygundur.",
        "OTOMATIK_RET": "Öneri: ret kararının korunması; başvurana itiraz hakkının hatırlatılması.",
        "UZMAN_INCELEMESI": "Öneri: yetki matrisine göre yetkili onayı ile, gerekirse tutar/vade düzenlemesiyle değerlendirilmesi.",
    }.get(outcome, "Öneri: yetkili kredi personelinin kararına sunulur.")
    return sections, recommendation


class UnderwriterAgent:
    def __init__(
        self,
        session: Session,
        application: Application,
        llm: LLMService | None = None,
        actor: str = "ajan",
    ) -> None:
        self.session = session
        self.app = application
        self.llm = llm or LLMService()
        self.tools = UnderwriterTools(session, application)
        self.actor = actor
        self.observations: dict[str, dict[str, Any]] = {}
        self._demo_index = 0

    # ------------------------------------------------------------ nodes
    async def _plan(self, state: AgentState) -> AgentState:
        steps = state.get("steps", [])
        if len(steps) >= MAX_STEPS:
            return {**state, "pending": None, "done": True}
        if self.llm.mode == "demo":
            if self._demo_index >= len(DEMO_PLAN):
                return {**state, "pending": None, "done": True}
            action = DEMO_PLAN[self._demo_index]
            self._demo_index += 1
            return {**state, "pending": action, "done": False, "protocol": "deterministic"}
        messages = state["messages"]
        protocol = state.get("protocol", "tools")
        try:
            if protocol == "tools":
                response: LLMResponse = await self.llm.provider.chat(messages, tools=openai_tools())
            else:
                response = await self.llm.provider.chat(messages)
        except ToolsNotSupportedError:
            react = [
                *messages,
                {
                    "role": "system",
                    "content": _REACT_HINT + ", ".join(t["name"] for t in TOOL_SPECS),
                },
            ]
            return await self._plan({**state, "messages": react, "protocol": "react"})
        except LLMCallError:
            return {**state, "pending": None, "done": True}
        if protocol == "tools" and response.tool_calls:
            call = response.tool_calls[0]
            if call.name == "draft_memo":
                return {**state, "pending": None, "done": True}
            return {**state, "pending": (call.name, call.arguments), "done": False}
        parsed = _parse_react(response.content)
        if parsed is None or parsed[0] == "draft_memo":
            return {**state, "pending": None, "done": True, "protocol": protocol}
        return {**state, "pending": parsed, "done": False, "protocol": "react"}

    async def _act(self, state: AgentState) -> AgentState:
        pending = state.get("pending")
        if not pending:
            return state
        name, args = pending
        result = self.tools.call(name, args)
        self.observations[name] = result
        step = AgentStep(
            step=len(state.get("steps", [])) + 1,
            tool=name,
            arguments=args,
            summary=_summarise(result),
        )
        append_audit(
            self.session,
            actor=self.actor,
            action="AGENT_STEP",
            entity_type="application",
            entity_id=self.app.id,
            payload={"step": step.step, "tool": name, "arguments": args},
        )
        observation = json.dumps(result, ensure_ascii=False, default=str)[:4000]
        messages = [
            *state.get("messages", []),
            {"role": "user", "content": f"[{name} sonucu]\n{observation}"},
        ]
        return {
            **state,
            "steps": [*state.get("steps", []), step],
            "messages": messages,
            "pending": None,
        }

    def _route(self, state: AgentState) -> str:
        return "draft" if state.get("done") else "act"

    def build(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("act", self._act)
        graph.set_entry_point("plan")
        graph.add_conditional_edges("plan", self._route, {"act": "act", "draft": END})
        graph.add_edge("act", "plan")
        return graph.compile()

    # ------------------------------------------------------------ run
    async def run(self) -> CreditMemo:
        initial: AgentState = {
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": f"Başvuru {self.app.id} için kredi tahsis memorandumu hazırla.",
                },
            ],
            "steps": [],
            "pending": None,
            "done": False,
            "protocol": "tools",
        }
        final: AgentState = await self.build().ainvoke(
            initial, {"recursion_limit": 2 * MAX_STEPS + 4}
        )
        # Guarantee the facts the memo needs even if a live model skipped tools.
        for name, args in DEMO_PLAN:
            if name not in self.observations:
                self.observations[name] = self.tools.call(name, args)
        return await self._draft(final)

    async def _draft(self, state: AgentState) -> CreditMemo:
        facts = self.tools.facts
        sections, recommendation = render_memo_sections(self.app, facts, self.observations)
        fallback_text = (
            "\n\n".join(f"## {s.title}\n{s.text}" for s in sections)
            + f"\n\n## Öneri\n{recommendation}"
        )
        fact_values = {fid: f.value for fid, f in facts.items()}
        context = json.dumps(
            {
                "alanlar": [
                    {"field_id": f.id, "etiket": f.label, "deger": f.value, "birim": f.unit}
                    for f in facts.values()
                ],
                "gozlemler": self.observations,
            },
            ensure_ascii=False,
            default=str,
        )[:12000]
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": "Aşağıdaki gözlemlerle 'Kredi Tahsis Memorandumu' yaz. Bölümler: Başvuran Profili ve Talep, Gelir ve "
                "İstihdam Doğrulaması, Borçluluk, Nakit Akışı, Belge Bulguları, Model ve Kural Sonuçları, Riskler ve "
                "Azaltıcılar, Fiyatlama, Öneri. Her bölüm '## Başlık' ile başlasın. Her sayının arkasına [f:alan_id] atfı "
                "koy; bağlamda olmayan sayı yazma. Karar verme, öneri yaz.\n\n" + context,
            },
        ]
        generation: Any = await self.llm.generate(
            "credit_memo",
            messages,
            fallback=lambda: fallback_text,
            guard=lambda text: validate_cited_text(text, fact_values).errors,
            redactor=Redactor(),
            application_id=self.app.id,
        )
        parsed_sections, parsed_recommendation = _split_sections(generation.text)
        return CreditMemo(
            application_id=self.app.id,
            sections=parsed_sections or sections,
            recommendation=parsed_recommendation or recommendation,
            citations=sorted(set(re.findall(r"\[(f:[A-Za-z0-9_.]+)\]", generation.text))),
            steps=state.get("steps", []),
            mode=generation.mode,
            protocol=state.get("protocol", "tools"),
            error_kind=generation.error_kind,
        )


def _split_sections(text: str) -> tuple[list[MemoSection], str]:
    sections: list[MemoSection] = []
    recommendation = ""
    for block in re.split(r"\n(?=## )", "\n" + text.strip()):
        block = block.strip()
        if not block.startswith("## "):
            continue
        title, _, body = block[3:].partition("\n")
        if title.strip().lower().startswith("öneri"):
            recommendation = body.strip()
        else:
            sections.append(MemoSection(title=title.strip(), text=body.strip()))
    return sections, recommendation
