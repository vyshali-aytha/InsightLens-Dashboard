# Module 7: Payment Failure Risk Scoring

Predicts the likelihood that a payment will fail, so a review team can focus
on the riskiest transactions instead of checking everything. Scoped honestly
as **payment failure risk**, not fraud detection — the schema has
`payment_method`/`payment_provider`/`amount`/customer & store history, but
no device fingerprint, IP/geolocation, or transaction-velocity data that
real fraud detection relies on.

## Project layout

| File | Role |
|---|---|
| `generate_payment_labels.py` | One-time / occasional: regenerates realistic, learnable `payment_status` labels from customer/store/provider/amount/seasonal risk factors. Not part of the regular run — only re-run this if the synthetic data itself needs refreshing. |
| `module7_features.py` | Step 1–3: pulls `dim_payment` + `fact_sales` (DB or local files), builds the label, engineers features, splits chronologically. |
| `module7_train.py` | Step 4–5: fits a calibrated Logistic Regression, evaluates it, saves `module7_model.joblib`. |
| `module7_score.py` | Step 6: scores current payments, writes per-transaction and dashboard-aggregate CSVs. |
| `module7_api.py` | Wraps the above in a FastAPI service so a dashboard or teammate's script can pull results over HTTP instead of running the CLI. |
| `db_utils.py` | Shared Postgres connection helper, reads `db_config.json`. |
| `db_config.json` | **Template only** — fill in your real credentials locally; never commit the filled-in version. |

## Running it — the normal sequence

```bash
pip install -r requirements.txt

# Steps 1-3: extract + engineer features from the live warehouse
python module7_features.py --db --out module7_features.csv

# Steps 4-5: train + evaluate (calibrated Logistic Regression)
python module7_train.py --train module7_features_train.csv --test module7_features_test.csv --model-out module7_model.joblib

# Step 6: score current payments, CSV output only by default
python module7_score.py --model module7_model.joblib --db --out payment_risk_scores.csv
```

This produces 3 CSVs:
- `payment_risk_scores.csv` — per-transaction `risk_score`, `rank_pct`, `flagged`, `month`, `is_peak_season`
- `payment_risk_scores_by_method_provider.csv` — dashboard: risk by payment method × provider, with both the model's `mean_risk_score` and the ground-truth `actual_fail_rate`
- `payment_risk_scores_by_month.csv` — dashboard: risk by calendar month, same model-vs-ground-truth pairing, for the seasonality finding below

Add `--persist-db` to also upsert scores into `predictions.payment_risk_scores` in Postgres (off by default — CSV only unless asked for).

## Running the API instead of the CLI

```bash
uvicorn module7_api:app --host 0.0.0.0 --port 8007 --reload
```

Then:
- `GET /health` — service status, rows scored, last refresh time
- `GET /scores?flagged_only=true&limit=50` — per-transaction risk scores
- `GET /charts/by_method_provider` / `GET /charts/by_month` — dashboard aggregates as JSON
- `POST /refresh` — force an immediate re-score instead of waiting for the background timer

The API loads the model once at startup and re-scores on a background timer
(`MODULE7_REFRESH_SECONDS`, default 300s) rather than re-running the full
pipeline on every request.

**Startup safety check:** before serving anything, the API verifies the
loaded `module7_model.joblib`'s feature set matches what the current code
expects, and refuses to start with a clear error if they don't match. This
exists because of a real incident during development — a model trained by
an older version of `module7_train.py` (missing `is_peak_season`, no
calibration) got deployed and served silently: it produced numbers that
looked plausible (in `[0,1]`) but were badly miscalibrated (mean
`risk_score` ~0.41 against an actual ~4% base rate), because nothing
caught that the model had literally never seen one of the features the
scoring code was now sending it. If you ever see this error at startup,
retrain in this project (`module7_train.py`) and copy the new
`module7_model.joblib` over the old one before restarting the API.

## Modeling approach, briefly

- **Logistic Regression**, `class_weight='balanced'` to counter the ~4%
  failure rate, chosen over Random Forest/XGBoost after a head-to-head
  comparison showed it nearly matched them (61% vs 63% of failures caught
  reviewing the riskiest 20%) while being far easier to explain.
- **Wrapped in `CalibratedClassifierCV`** (sigmoid/Platt scaling, 5-fold)
  so `risk_score` is a real, checkable probability — among payments scored
  ~0.6, roughly 60% should actually fail — rather than just a ranking
  signal distorted by the balanced class weighting.
- **Chronological train/test split** (not random) so evaluation mirrors
  real deployment: train only sees the past, test only the future.
- **Leakage-safe behavioral features**: `customer_prior_fail_rate`,
  `store_prior_fail_rate`, etc. are expanding windows computed strictly
  before each transaction's own date.
- **Evaluated primarily by capacity-based precision/recall** (reviewing
  the riskiest 5/10/20% of transactions) since that's how the score is
  actually consumed operationally — a fixed 0.5 probability cutoff is kept
  only as a secondary diagnostic, clearly labeled as such in the training
  log.

## The seasonality finding (Oct/Nov/Dec)

A reviewer check found that monthly transaction volume correlates with
monthly failure rate (0.58) in the original data, with Oct/Nov/Dec
consistently the highest-volume, higher-failure months — a holiday-season
pattern. This is now:
- **A real, deliberate signal in the data itself** (`generate_payment_labels.py`
  adds a documented logit bump for these three months, verified at
  z ≈ 6.2, p < 0.00001 — not incidental noise).
- **A model feature** (`is_peak_season` in `module7_features.py`, fed to
  the model in `module7_train.py`).
- **Directly visible on the dashboard** via `actual_fail_rate` (ground
  truth, independent of the model) alongside `mean_risk_score` (the
  model's own read) in `payment_risk_scores_by_month.csv` — reported side
  by side specifically so a noisy training run's `mean_risk_score` alone
  is never the only evidence for this pattern.
- **Confirmed via the model's own learned coefficient**: `module7_train.py`
  logs an `is_peak_season net effect` line straight from the fitted
  Logistic Regression weights, independent of any month-level aggregation
  confounding.

## Known limitations (stated honestly, not hidden)

- No device fingerprint, IP/geolocation, or velocity data — this predicts
  **failure risk**, not fraud, per the scoping note above.
- PR-AUC (~0.20) indicates real but modest signal — there's a ceiling set
  by the available fields, not by any remaining modeling choice.
- Small-sample categories (e.g. a payment provider with only ~50
  transactions) are more prone to overfit coefficients than high-volume
  ones (e.g. BHIM at 1,000+) — treat `actual_fail_rate`/`mean_risk_score`
  for low-`n_resolved` rows in the dashboard CSVs with proportionally more
  caution.
