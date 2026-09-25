# Compliance Notes (short, sourced; not legal advice)

The platform was **designed with these requirements in mind**; it is not certified or audited as
compliant with any of them. Every citation below was checked against an official source on
2026-09-25 (table at the end); anything that could not be confirmed is marked as such.

## BDDK — credit operations and allocation
*Bankaların Kredi İşlemlerine İlişkin Yönetmelik* and BDDK allocation/monitoring expectations:
decisions rest on documented repayment capacity and are taken by authorised organs under written
policy. → Versioned policy (`rules/`), authority matrix and four-eyes (`rules/authority_matrix.yaml`),
credit memorandum with cited figures, early-warning watchlist, hash-chained audit trail and
replayable decisions.

## BDDK — information systems and data localisation
*Bankaların Bilgi Sistemleri ve Elektronik Bankacılık Hizmetleri Hakkında Yönetmelik* requires
primary/secondary systems in Turkey and managed outsourcing. → OpenAI-compatible LLM layer that can
point to an on-premise model with `LLM_BASE_URL`; PII pseudonymisation before any external call;
circuit breakers and retries around external providers (ADR 0006).

## KVKK (Law No. 6698)
* **Art. 5 / explicit consent:** information notice and consents captured per application; every
  data-source call is gated by a valid, unexpired consent (`consents` table).
* **Art. 11 (1)(g):** right to object to a result arising exclusively from automated processing →
  adverse letters include reason codes and the objection route; `POST /applications/{id}/objection`
  forces human review (`ITIRAZ_INCELEMESI`).
* **Art. 13:** requests are answered within 30 days → 720-hour SLA for objections.
* **Art. 12 / Board decision 2018/10 (adequate measures for special categories):** Fernet encryption
  of PII, blind indexes, log masking, role-based access, retention/anonymisation job.
* **Board decision 2020/173 (27 Feb 2020):** consent cannot be made a condition of the service →
  open-banking consent is optional; the decision runs without cash-flow data when it is refused.

## EU AI Act (Regulation (EU) 2024/1689)
Creditworthiness assessment of natural persons is **high-risk (Annex III, point 5(b))**. The original
application date of the Annex III obligations (2 August 2026) was postponed to **2 December 2027** by
the Digital Omnibus on AI, reported in force on 27 July 2026 (secondary sources; the Official Journal
text was not re-checked here). Mapped controls: risk management (model inventory, drift, fairness),
data governance (real-data validation, excluded protected attributes), technical documentation
(model card, ADRs, validation report), logging (audit chain, `llm_calls`), transparency (reason
codes, letters), human oversight (workbench, four-eyes, objections), accuracy/robustness (metrics,
replay tests, fallback modes).

## US reference points
* **SR 26-2 / OCC Bulletin 2026-13 (17 April 2026), Revised Guidance on Model Risk Management** — it
  supersedes SR 11-7 and SR 21-8. The v1 audit suspected this citation was invented; it is real.
  Mapped: model inventory, validation on real data, ongoing monitoring, evidence-based challenger
  promotion.
* **ECOA / Regulation B (12 CFR 1002.9):** specific principal reasons for adverse action → applicant
  specific, material reason codes. **CFPB Circular 2022-03** (complex algorithms) was **withdrawn on
  12 May 2025**; it is cited only as history. The status of Circular 2023-03 was not re-verified.

## Taxes and pricing parameters
BSMV and KKDF rates and the TCMB maximum contractual rate in `rules/pricing.yaml` are **examples** —
verify against the current Resmî Gazete / TCMB announcements before use. The Basel IRB "other retail"
correlation formula follows the BCBS risk-weight functions.

## Verification (2026-09-25)

| Citation | Status | Source |
|---|---|---|
| BDDK Kredi İşlemleri Yönetmeliği | verified (title; amended 2023) | <https://www.bddk.org.tr/Duyuru/Detay/671> |
| BDDK Bilgi Sistemleri ve Elektronik Bankacılık Yönetmeliği | verified (title) | <https://www.bddk.org.tr/Mevzuat/Liste/134> |
| KVKK Law 6698 | law number verified; page renders client-side, text not re-read | <https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=6698&MevzuatTur=1&MevzuatTertip=5> |
| KVKK Board decision 2018/10 | verified (title) | <https://www.kvkk.gov.tr/Icerik/7201/ilke-kararlari> |
| KVKK Board decision 2020/173 | verified | <https://www.kvkk.gov.tr/Icerik/6739/2020-173> |
| EU AI Act, Annex III 5(b) | verified | <https://eur-lex.europa.eu/eli/reg/2024/1689/oj> |
| AI Act Annex III date → 2 Dec 2027 | secondary sources only | <https://www.regulation-ai.eu/en/annex-iii/> |
| SR 26-2 | verified | <https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm> |
| OCC Bulletin 2026-13 | verified | <https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html> |
| CFPB Circular 2022-03 | verified — **withdrawn 12 May 2025** | <https://www.federalregister.gov/documents/2025/05/12/2025-08286/interpretive-rules-policy-statements-and-advisory-opinions-withdrawal> |
| CFPB Circular 2023-03 | published; current status **not verified** | <https://www.consumerfinance.gov/compliance/circulars/circular-2023-03-adverse-action-notification-requirements-and-the-proper-use-of-the-cfpbs-sample-forms-provided-in-regulation-b/> |
| Basel IRB other-retail correlation | document verified (BCBS explanatory note, 2005); formula as in the IRB framework | <https://www.bis.org/bcbs/irbriskweight.pdf> |
