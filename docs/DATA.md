# Data

The platform uses two kinds of data:

1. **Synthetic Turkish applications** (`app/decisioning/training.py`) for the production model,
   because its inputs — KKB score, DSR, open-banking cash-flow features — do not exist in any public
   dataset.
2. **Real public credit-default data** to validate the methodology (lane A), to train the
   behaviour sub-score and to anchor the PD level of the production model (lane B). The anchoring
   imposes the level from a real proxy curve; it is not a validation of that level.

## Public datasets

`scripts/fetch_public_credit_data.py` is the only component that goes to the network. It downloads
each archive, verifies its SHA-256 against the value pinned in the script, and writes a normalised
CSV to `data/external/<set>/data.csv` (the whole `data/` directory is gitignored) plus
`data/external/manifest.json`.

| Set | Rows | Default rate | Licence | Source |
|---|---|---|---|---|
| `uci_taiwan` — Default of Credit Card Clients | 30,000 | 22.12 % | CC BY 4.0 | <https://archive.ics.uci.edu/dataset/350> |
| `german_credit` — Statlog (German Credit Data) | 1,000 | 30.00 % | CC BY 4.0 | <https://archive.ics.uci.edu/dataset/144> |

Checksums (SHA-256):

| File | SHA-256 |
|---|---|
| `uci_taiwan` archive (`default+of+credit+card+clients.zip`) | `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602` |
| `uci_taiwan` normalised `data.csv` | `e28803eec99215182faffbced63edd940d5034ed240443084ae72d254f147f89` |
| `german_credit` archive (`statlog+german+credit+data.zip`) | `e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b` |
| `german_credit` normalised `data.csv` | `2a98ed3725ae530f43792c54870d7a3a424953f6baec5dcc707c8135a93a1f80` |
| `tests/fixtures/uci_taiwan_sample.csv` | `603a4c51260b0c02d6b3b4c75a29f698c761cc966f00b5d746152f4864d2b479` |

Attribution:

* Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C55S3H>. Licensed under CC BY 4.0. Original study: Yeh, I. C. & Lien,
  C. H. (2009), "The comparisons of data mining techniques for the predictive accuracy of
  probability of default of credit card clients", *Expert Systems with Applications* 36(2).
* Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5NC77>. Licensed under CC BY 4.0.

Normalisation: the Taiwan `ID` column is dropped and `default payment next month` is renamed to
`default`; German Credit gets readable column names and `default = 1` for class 2 ("bad").

**Kaggle sets** (Home Credit, Give Me Some Credit, Lending Club) are optional. The fetcher only
checks whether `KAGGLE_USERNAME` and `KAGGLE_KEY` exist in the environment; it never reads or
prints their values. In the run recorded here they were **not** defined, so these sets were
**skipped**.

### Offline fixture

`tests/fixtures/uci_taiwan_sample.csv` is a 3,000-row sample of the Taiwan set, stratified by
target and `SEX` (seed `20260925`), 262 KB, with the CC BY 4.0 attribution in its first lines
(read it with `pandas.read_csv(path, comment="#")`). Tests that need the full data are marked
`@pytest.mark.external_data` and are skipped when `data/external/` is absent. Pre-computed
validation metrics are committed under `artifacts/validation/`, so the application and the test
suite run without the network.

## Taiwan set: timing of the variables and leakage

The target is default on the payment due in **October 2005** ("default payment next month"). All
predictors describe April–September 2005:

| Variable | Month | Meaning |
|---|---|---|
| `PAY_0` | September 2005 | repayment status (−2 no consumption, −1 paid in full, 0 revolving credit used, 1–8 months of delay) |
| `PAY_2` … `PAY_6` | August … April 2005 | same, earlier months (there is no `PAY_1`) |
| `BILL_AMT1` … `BILL_AMT6` | September … April 2005 | statement balance |
| `PAY_AMT1` … `PAY_AMT6` | September … April 2005 | amount paid in that month |

`PAY_0` is the status of the **last observed month before** the target month, i.e. information that
is available at decision time — it is the most recent bureau-style arrears flag, not the outcome.
No variable describes October 2005, so there is no same-month leakage. `PAY_0` is nevertheless by
far the strongest predictor (default rate 13 % at status 0 vs. 69 % at a two-month delay), which is
why lane A reports both the full model and its stability across folds.

Protected attributes used **only** for fairness analysis (never as model inputs): `SEX`
(1 = male, 2 = female), `AGE` (banded), `EDUCATION`, `MARRIAGE`. Undocumented codes (`EDUCATION`
0/5/6, `MARRIAGE` 0) are grouped as "other/unknown".

## Semantic mapping (lane B)

Implemented in `app/decisioning/public_mapping.py`; results in `artifacts/validation/lane_b.json`
and the lane B section of `docs/VALIDATION_REPORT.md`.

| Platform feature | Public source (Taiwan) | Transformation | Rationale |
|---|---|---|---|
| `max_dpd_24m` → `delay_months` | `PAY_0 … PAY_6` | platform: ⌈max_dpd_24m / 30⌉ capped at 8; public: max(0, max PAY_x) | Both measure the worst arrears in months; KKB reports days past due |
| `delinquency_count_24m` → `delinquent_months` | `PAY_0 … PAY_6` | platform: count capped at 6; public: months with PAY_x > 0 | Frequency of arrears |
| `bureau_utilisation` → `utilisation` | `BILL_AMT1 / LIMIT_BAL` | clipped to [0, 1.5] | Revolving balance over limit, the standard bureau utilisation measure |
| `age` | `AGE` | unchanged | Eligibility rules and fairness monitoring only — never a model input |

`PAY_AMT / BILL_AMT` (payment ratio) has no counterpart in the platform's bureau payload, so it is
used in lane A but not mapped.

Known differences, stated rather than hidden:

* **Window.** The public history covers 6 months, KKB features cover 24 months.
* **Target.** Taiwan: default on the next monthly card payment. Platform: 90+ days past due within
  12 months on an instalment loan. The anchoring transfers the *shape* of the arrears→default curve
  and imposes the next-month card default rate as the level of a 12-month 90+DPD loan PD — a proxy
  level, not a validated Turkish PD.
* **Population.** The synthetic generator keeps a Turkish applicant mix (80 % without recent arrears
  vs. 66 % in the card book); only the default rate *within* each delinquency band is anchored.

The behaviour sub-score (`bureau_behavior_score`, artifact `artifacts/models/bureau_behavior_v1.*`)
is a monotone LightGBM on the three behaviour features, trained and isotonically calibrated on
real Taiwan defaults (hold-out AUC 0.740), and enters the production PD model as a feature.
