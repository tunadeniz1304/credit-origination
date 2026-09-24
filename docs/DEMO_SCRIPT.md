# 3-minute demo script

**0:00 — Start.** `docker compose up --build`, open `http://localhost:8000`. Point out the header badge **"AI: Canlı (deepseek-v4-flash)"** (or *Demo* without a key).

**0:15 — Load scenarios.** Log in as **admin** → *Yönetim panosu* → **Demo senaryosu yükle**. Six personas run through the real pipeline; the dashboard fills with automation rate, approval rate, PD distribution, AIR and drift.

**0:45 — Applicant view.** Log out, log in as **basvuran** → *Başvurularım*:
* *Elif Yıldırım* (clean): timeline, risk-based offer, instalment incl. BSMV/KKDF, annual cost rate → **Teklifi kabul et** → pre-contract form PDF.
* *Derya Şahin* (high DSR): conditional counter-offer + "Ne değişirse onaylanırsınız?" suggestions.

**1:20 — Workbench.** Log in as **uzman** → *Çalışma masası*:
* *Onur Çelik* (tampered payslip): *Belgeler* tab shows incremental save, editing-tool producer, foreign font, arithmetic mismatch.
* *Ali Tan* (ring): *Halka* tab graph of shared phone/IBAN.
* *Seda Arslan* (grey zone): *Karar + SHAP* waterfall, reason codes with points lost, **replay** ✓; *AI memorandum* with cited numbers; *Politika sor* → cited answer.

**2:20 — Four eyes.** As **komite** → *Karar ver* → approve the pending review created by *uzman* (the maker cannot approve their own request).

**2:40 — Governance.** As **modelyon** → dashboard: champion/challenger agreement, model card, AIR ≥ 0.8, PSI table; promotion needs a second model manager.

**3:00 — Close.** "The engine decides, people approve, the LLM only explains — every number is cited, every step is audited and replayable."
