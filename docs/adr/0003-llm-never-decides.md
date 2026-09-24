# ADR 0003 — The LLM never makes a binding decision

**Status:** accepted

## Context
LLMs are useful for memoranda, summaries and plain-language letters but are non-deterministic and can hallucinate. BDDK expects decisions by authorised organs; KVKK art. 11 and the EU AI Act require human oversight of automated adverse decisions.

## Decision
Outcome, limit and price come only from the deterministic engine or an authorised human (authority matrix, four-eyes). The LLM explains: a three-step chain (structured analysis → committee summary → applicant letter) and a supervised agent that proposes a *recommendation*. Every number in a narrative must cite a decision-context `field_id`; a citation/number guard rejects violations (one repair, then deterministic fallback). LLM failures never fail an application.

## Consequences
+ Narratives are auditable and cannot contradict the decision record.
+ Demo mode renders the same outputs deterministically from templates.
− Narratives are constrained; free-form insight beyond the context is not allowed (by design).
