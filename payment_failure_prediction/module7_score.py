"""
module7_score.py
-----------------
Module 7: Payment Failure Risk Scoring — Step 6 (persistence & delivery).

Loads the trained model (a scikit-learn Pipeline saved with joblib,
produced by module7_train.py) and scores payments, writing the result
to a new `predictions.payment_risk_scores` table in the warehouse so
the existing Streamlit dashboard can read it the same way it reads any
other table — no new integration pattern needed.

Usage:
    # score everything currently in dim_payment / fact_sales, write to Postgres
    python module7_score.py --model module7_model_random_forest.joblib --db

    # score a local export instead, write to a CSV (no DB needed)
    python module7_score.py --model module7_model_random_forest.joblib \\
        --sales sales.csv --payment payment.csv --out risk_scores.csv

Writes 3 CSVs per run:
  - <out>                       per-transaction risk_score/rank_pct/flagged
  - <out>_by_method_provider.csv   dashboard: risk by payment_method x provider
  - <out>_by_month.csv             dashboard: risk by calendar month (seasonality)
"""

import argparse
import logging

import joblib
import pandas as pd

from module7_features import build_features, build_label, load_from_db, load_from_files

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

CATEGORICAL_FEATURES = ["payment_method", "payment_provider", "is_peak_season"]
NUMERIC_FEATURES = [
    "log_amount",
    "customer_prior_fail_rate", "customer_prior_txn_count",
    "store_prior_fail_rate", "store_prior_txn_count",
]

CREATE_SCHEMA_SQL = "CREATE SCHEMA IF NOT EXISTS predictions;"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions.payment_risk_scores (
    payment_id      INTEGER PRIMARY KEY,
    transaction_id  BIGINT NOT NULL,
    risk_score      NUMERIC(6,5) NOT NULL,
    scored_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_name      VARCHAR(100) NOT NULL
);
"""

UPSERT_SQL = """
INSERT INTO predictions.payment_risk_scores (payment_id, transaction_id, risk_score, model_name)
VALUES (%s, %s, %s, %s)
ON CONFLICT (payment_id) DO UPDATE
    SET risk_score = EXCLUDED.risk_score,
        scored_at  = CURRENT_TIMESTAMP,
        model_name = EXCLUDED.model_name;
"""


def score(model, df):
    """
    Applies the same feature-building logic used at training time
    (build_features / cold-start handling) so scoring is consistent
    with how the model was trained, then returns risk scores.

    Cold-start note: rows with no prior history get NaN
    customer/store_prior_fail_rate here; the trained Pipeline's
    OneHotEncoder/StandardScaler can't take NaN, so we fill with the
    same population-average logic module7_features.build_time_split
    uses at training time — computed once here from this same scoring
    batch, since at inference time there's no "train" set to borrow
    the constant from. For small scoring batches, consider persisting
    the training-time constant instead of recomputing it live.
    """
    df = df.copy()
    pop_rate = df["label"].mean() if "label" in df.columns else 0.0
    for col in ["customer_prior_fail_rate", "store_prior_fail_rate"]:
        df[col] = df[col].fillna(pop_rate)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    df["risk_score"] = model.predict_proba(X)[:, 1]
    return df


def write_to_db(scored_df, model_name):
    import db_utils
    conn = db_utils.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_SCHEMA_SQL)
            cur.execute(CREATE_TABLE_SQL)
            rows = [
                (int(r.payment_id), int(r.transaction_id), float(r.risk_score), model_name)
                for r in scored_df.itertuples()
            ]
            cur.executemany(UPSERT_SQL, rows)
        conn.commit()
        logger.info("write_to_db: upserted %d row(s) into predictions.payment_risk_scores", len(rows))
    finally:
        conn.close()


LOW_SAMPLE_THRESHOLD = 100  # below this many resolved transactions, a group's
                             # actual_fail_rate/mean_risk_score is statistically
                             # thin — e.g. one or two extra failures can swing
                             # the rate by several points. Flagged, not hidden.


def _add_actual_fail_rate(summary, scored_df, group_cols):
    """
    Adds a ground-truth actual_fail_rate column alongside the model's
    mean_risk_score, computed directly from payment_status — no model
    involved. PENDING rows are excluded from the denominator (unresolved
    outcome, same rule build_label() uses for training).

    This exists because mean_risk_score can be noisy/underweighted on any
    single training run (regularization, calibration fold variance, class
    imbalance) even when a real historical pattern exists — see the
    is_peak_season case, where the raw data shows a clear Oct/Nov/Dec
    effect but a given model run may not surface it cleanly in its own
    aggregate scores. actual_fail_rate is the direct, model-independent
    answer to "did failures really happen more often here historically".

    Also adds low_sample (n_resolved < LOW_SAMPLE_THRESHOLD): a flag, not
    a fix — small categories (e.g. a payment provider with only ~50-60
    resolved transactions) produce noisier rates than high-volume ones,
    so this exists to tell a dashboard reader which rows to sanity-check
    further, rather than silently presenting every row at equal
    confidence regardless of how much data backs it.
    """
    resolved = scored_df[scored_df["payment_status"].isin(["SUCCESS", "FAILED"])].copy()
    resolved["is_failed"] = (resolved["payment_status"] == "FAILED").astype(int)
    actual = (
        resolved.groupby(group_cols)
        .agg(n_resolved=("is_failed", "size"), actual_fail_rate=("is_failed", "mean"))
        .reset_index()
    )
    actual["low_sample"] = actual["n_resolved"] < LOW_SAMPLE_THRESHOLD
    return summary.merge(actual, on=group_cols, how="left")


def summarize_by_method_provider(scored_df):
    """
    Dashboard output: aggregates the per-transaction scores up to the
    payment_method x payment_provider level, per the spec's 'flagged
    high-risk payment method/provider combinations for the dashboard'
    requirement. Complements the per-transaction risk_score/flagged
    output — this answers "which combos are riskiest overall", not
    "which individual transactions to review". Includes both the
    model's mean_risk_score and the ground-truth actual_fail_rate side
    by side — see _add_actual_fail_rate for why both matter.
    """
    summary = (
        scored_df.groupby(["payment_method", "payment_provider"])
        .agg(
            n_transactions=("risk_score", "size"),
            n_flagged=("flagged", "sum"),
            mean_risk_score=("risk_score", "mean"),
        )
        .reset_index()
    )
    summary["flagged_rate"] = summary["n_flagged"] / summary["n_transactions"]
    summary = _add_actual_fail_rate(summary, scored_df, ["payment_method", "payment_provider"])
    summary = summary.sort_values("mean_risk_score", ascending=False).reset_index(drop=True)
    return summary


def summarize_by_month(scored_df):
    """
    Dashboard output: aggregates risk by calendar month, so the
    seasonality pattern that motivated is_peak_season (monthly volume
    correlates with monthly failure rate at 0.58 in this data, with
    Oct/Nov/Dec consistently elevated) is directly visible on the
    dashboard, not just baked invisibly into the model. Includes both
    mean_risk_score (model) and actual_fail_rate (ground truth) — see
    _add_actual_fail_rate for why both are reported, not just one.
    """
    summary = (
        scored_df.groupby("month")
        .agg(
            n_transactions=("risk_score", "size"),
            n_flagged=("flagged", "sum"),
            mean_risk_score=("risk_score", "mean"),
        )
        .reset_index()
    )
    summary = _add_actual_fail_rate(summary, scored_df, ["month"])
    summary["flagged_rate"] = summary["n_flagged"] / summary["n_transactions"]
    summary["is_peak_season"] = summary["month"].isin([10, 11, 12])
    summary = summary.sort_values("month").reset_index(drop=True)
    return summary


def run(model_path, use_db=False, sales_path=None, payment_path=None, out_path=None, persist_db=False, capacity_pct=10.0):
    model = joblib.load(model_path)
    model_name = model_path

    if use_db:
        raw = load_from_db()
    else:
        raw = load_from_files(sales_path, payment_path)

    # Reuse build_label only to get transaction_date/features consistent
    # with training; keep PENDING/FAILED/SUCCESS all in for scoring
    # (unlike training, we score every payment, not just resolved ones).
    raw["label"] = (raw["payment_status"] == "FAILED").astype(int)  # placeholder, only used for pop_rate fallback
    featured = build_features(raw)

    scored = score(model, featured)

    # Capacity-based flagging: rank all scored payments by risk_score
    # (highest risk first) and flag the top `capacity_pct`% for review.
    # This is what ops actually reviews — a ranked list, not a 0.5
    # probability cutoff. rank_pct=0 is the single highest-risk payment.
    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    scored["rank_pct"] = (scored.index + 1) / len(scored)
    scored["flagged"] = scored["rank_pct"] <= (capacity_pct / 100.0)
    n_flagged = int(scored["flagged"].sum())
    logger.info("Flagged top %.1f%% by risk_score = %d of %d payments for review",
                capacity_pct, n_flagged, len(scored))

    out_cols = ["payment_id", "transaction_id", "risk_score", "rank_pct", "flagged", "month", "is_peak_season"]

    # Output is always CSV — write_to_db() is still available above but is
    # opt-in only (--persist-db), never implied by --db. --db here controls
    # the READ source only.
    scored[out_cols].to_csv(out_path, index=False)
    logger.info("Wrote %d scored rows to %s", len(scored), out_path)

    # Dashboard summary: risk aggregated by payment_method x payment_provider.
    summary_path = out_path.replace(".csv", "_by_method_provider.csv")
    summary = summarize_by_method_provider(scored)
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %d method/provider combo(s) to %s", len(summary), summary_path)
    logger.info("Top risky combos:\n%s",
                summary.head(5).to_string(index=False))

    # Dashboard summary: risk by calendar month (seasonality check).
    month_summary_path = out_path.replace(".csv", "_by_month.csv")
    month_summary = summarize_by_month(scored)
    month_summary.to_csv(month_summary_path, index=False)
    logger.info("Wrote %d month(s) to %s", len(month_summary), month_summary_path)
    logger.info("Risk by month:\n%s", month_summary.to_string(index=False))

    if persist_db:
        write_to_db(scored[["payment_id", "transaction_id", "risk_score"]], model_name)

    return scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 7: Step 6 (score)")
    parser.add_argument("--model", required=True, help="Path to the .joblib model from module7_train.py")
    parser.add_argument("--db", action="store_true", help="Read payments from the live Postgres warehouse (input source only)")
    parser.add_argument("--sales", help="Local sales export (csv/xlsx) — used when not --db")
    parser.add_argument("--payment", help="Local payment export (csv/xlsx) — used when not --db")
    parser.add_argument("--out", default="payment_risk_scores.csv", help="CSV output path — always used, regardless of --db")
    parser.add_argument("--persist-db", action="store_true", help="Also upsert scores into predictions.payment_risk_scores (off by default)")
    parser.add_argument("--capacity-pct", type=float, default=10.0, help="Flag the top N%% of payments by risk_score for review (default 10)")
    args = parser.parse_args()

    run(args.model, use_db=args.db, sales_path=args.sales, payment_path=args.payment,
        out_path=args.out, persist_db=args.persist_db, capacity_pct=args.capacity_pct)
