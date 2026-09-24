# Compliance Notes (short, sourced; not legal advice)

## BDDK — credit operations and allocation
*Bankaların Kredi İşlemlerine İlişkin Yönetmelik* and BDDK allocation/monitoring expectations: decisions rest on documented repayment capacity and are taken by authorised organs under written policy. → Versioned policy (`rules/`), authority matrix and four-eyes (`rules/authority_matrix.yaml`), credit memorandum with cited figures, early-warning watchlist, hash-chained audit trail and replayable decisions.

## BDDK — information systems and data localisation
*Bankaların Bilgi Sistemleri ve Elektronik Bankacılık Hizmetleri Hakkında Yönetmelik* requires primary/secondary systems in Turkey and managed outsourcing. → OpenAI-compatible LLM layer that can point to an on-premise model with `LLM_BASE_URL`; PII pseudonymisation before any external call; circuit breakers and retries around external providers (ADR 0006).

## KVKK (Law No. 6698)
* **Art. 10 / explicit consent:** information notice and consents captured per application; every data-source call is gated by a valid, unexpired consent (`consents` table).
* **Art. 11 (1)(g):** right to object to a result arising exclusively from automated processing → adverse letters include reason codes and the objection route; `POST /applications/{id}/objection` forces human review (`ITIRAZ_INCELEMESI`).
* **Art. 13:** requests are answered within 30 days → 720-hour SLA for objections.
* **Art. 12 / Board decision on adequate measures (2018/10):** Fernet encryption of PII, blind indexes, log masking, role-based access, retention/anonymisation job.
* The KVKK Board's guidance on automated decision-making and profiling (e.g., the 2020/173 decision context on explicit consent) informed the consent design.

## EU AI Act
Creditworthiness assessment of natural persons is **high-risk (Annex III, point 5(b))**; high-risk obligations apply from **2 December 2027** under the current timetable (check the latest implementing acts). Mapped controls: risk management (model inventory, drift, fairness), data governance (synthetic data documentation, excluded protected attributes), technical documentation (model card, ADRs), logging (audit chain, `llm_calls`), transparency (reason codes, letters), human oversight (workbench, four-eyes, objections), accuracy/robustness (metrics, replay tests, fallback modes).

## US reference points
* **SR 26-2** (successor to SR 11-7, model risk management): model inventory, validation, ongoing monitoring, challenger models.
* **CFPB / ECOA Regulation B:** specific principal reasons for adverse action; the applicant-specific reason codes follow this expectation rather than generic statements.

## Taxes and pricing parameters
BSMV and KKDF rates and the TCMB maximum contractual rate in `rules/pricing.yaml` are **examples** — verify against the current Resmî Gazete / TCMB announcements before use.
