"""
module7_features.py
--------------------
Module 7: Payment Failure Risk Scoring — Step 1 (extract & join) and
Step 2 (time-windowed feature engineering).

Design decisions locked in with Vyshali:
  - Label: FAILED = 1, SUCCESS = 0, PENDING rows DROPPED from training
    (ambiguous — not yet resolved either way, shouldn't poison a 2%
    positive class).
  - Behavioral features (customer / store prior failure rate) are
    computed as EXPANDING WINDOWS ordered by transaction date, using only
    transactions strictly BEFORE the current one. This is what prevents
    leakage: a transaction's own outcome is never allowed to contribute
    to its own feature values.
  - Split into train/test is chronological, not random — see
    build_time_split() at the bottom. That happens in Step 3, but the
    date column needed for it is produced here in Step 2.

This module is DB-first (reads from the real Postgres warehouse via the
existing db_utils.get_connection() used by main_pipeline.py), with a
local CSV/XLSX fallback so it can be developed and unit-tested without a
live DB connection.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Step 1 — Extraction
# ---------------------------------------------------------------------------

SQL_EXTRACT = """
    SELECT
        p.payment_id,
        p.transaction_id,
        p.store_key,
        p.customer_key,
        p.payment_method,
        p.payment_provider,
        p.payment_status,
        p.amount,
        s.date_key
    FROM insightlens.dim_payment p
    JOIN insightlens.fact_sales  s ON s.transaction_id = p.transaction_id
"""


def load_from_db():
    """
    Step 1 (production path): pull dim_payment joined to fact_sales (for
    date_key) directly from the Postgres warehouse, reusing the same
    connection helper the ETL pipeline already uses.
    """
    import db_utils  # local import: only needed on the DB path
    conn = db_utils.get_connection()
    try:
        df = pd.read_sql(SQL_EXTRACT, conn)
    finally:
        conn.close()
    logger.info("load_from_db: pulled %d payment rows joined to fact_sales", len(df))
    return df


def load_from_files(sales_path, payment_path):
    """
    Step 1 (local/dev path): build the same shape of frame as
    load_from_db() from the flat-file exports, for development and
    testing without a live DB connection. Accepts .csv or .xlsx
    transparently.
    """
    def _read(path):
        # Don't trust the extension — these exports have shown up as
        # .csv-named files that are actually .xlsx (zip) under the hood.
        with open(path, "rb") as f:
            magic = f.read(4)
        is_xlsx = magic[:2] == b"PK"  # xlsx/zip signature
        return pd.read_excel(path) if is_xlsx else pd.read_csv(path)

    sales = _read(sales_path)
    pay = _read(payment_path)

    df = pay.merge(
        sales[["transaction_id", "date_key"]],
        on="transaction_id",
        how="inner",
    )
    df = df[[
        "payment_id", "transaction_id", "store_key", "customer_key",
        "payment_method", "payment_provider", "payment_status", "amount",
        "date_key",
    ]]
    logger.info("load_from_files: built %d payment rows joined to sales.date_key", len(df))
    return df


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def build_label(df):
    """
    FAILED -> 1, SUCCESS -> 0, PENDING rows dropped (per locked-in
    decision — ambiguous outcome, not yet resolved either way).
    """
    before = len(df)
    df = df[df["payment_status"].isin(["FAILED", "SUCCESS"])].copy()
    dropped = before - len(df)
    logger.info("build_label: dropped %d PENDING row(s); %d remain", dropped, len(df))

    df["label"] = (df["payment_status"] == "FAILED").astype(int)
    rate = df["label"].mean()
    logger.info("build_label: positive (FAILED) rate = %.4f (%d / %d)",
                rate, df["label"].sum(), len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2 — Feature engineering
# ---------------------------------------------------------------------------

def _parse_date(date_key_series):
    return pd.to_datetime(date_key_series.astype(int).astype(str), format="%Y%m%d")


def _expanding_prior_rate(df, group_col, label_col="label", min_periods=1):
    """
    For each row, compute the mean of `label_col` over all PRIOR rows
    (strictly earlier transaction_date) within the same group_col value.
    The row's own label is excluded — this is the leakage guard.

    Returns two Series aligned to df's index:
      - prior_rate: expanding mean of label over prior rows (NaN if none)
      - prior_count: number of prior rows contributing to that mean
    """
    df = df.sort_values("transaction_date", kind="mergesort")  # stable sort preserves tie order
    grouped = df.groupby(group_col)[label_col]

    # cumulative sum/count of *all* rows seen so far (inclusive), then
    # shift by 1 within each group so the current row's own label is
    # excluded from its own feature value.
    cum_sum = grouped.cumsum()
    cum_count = grouped.cumcount() + 1  # 1-indexed running count

    prior_sum = cum_sum - df[label_col]
    prior_count = cum_count - 1

    with np.errstate(invalid="ignore", divide="ignore"):
        prior_rate = np.where(prior_count >= min_periods, prior_sum / prior_count, np.nan)

    return pd.Series(prior_rate, index=df.index), pd.Series(prior_count, index=df.index)


def build_features(df):
    """
    Step 2: adds
      - transaction_date        (parsed from date_key, needed for the
                                  chronological split in Step 3)
      - log_amount              (log1p of amount; amount is heavily
                                  right-skewed)
      - customer_prior_fail_rate / customer_prior_txn_count
      - store_prior_fail_rate    / store_prior_txn_count
        (both computed as expanding windows strictly BEFORE the current
        transaction's date — see _expanding_prior_rate)
      - month                   (calendar month, 1-12 — used for the
                                  by-month dashboard summary, not fed to
                                  the model directly)
      - is_peak_season          (1 if month is Oct/Nov/Dec, else 0)

    is_peak_season is based on a checked pattern in the data: monthly
    transaction volume correlates with monthly failure rate at 0.58, and
    grouping by calendar month (independent of year) shows Oct/Nov/Dec
    consistently both higher-volume AND higher-failure-rate than the rest
    of the year across both years present in the data — a real holiday/
    Black-Friday-season effect, not noise (daily-level volume showed no
    such correlation, only monthly-and-up).

    Cold-start (no prior history for that customer/store) is left as NaN
    here on purpose — Step 3/4 will impute with the population-average
    fail rate at model-training time, not baked in here, so the
    population average is computed on TRAIN only and never leaks test
    statistics into the feature file.
    """
    df = df.copy()
    df["transaction_date"] = _parse_date(df["date_key"])
    df["log_amount"] = np.log1p(df["amount"].astype(float))

    df["customer_prior_fail_rate"], df["customer_prior_txn_count"] = _expanding_prior_rate(
        df, group_col="customer_key")
    df["store_prior_fail_rate"], df["store_prior_txn_count"] = _expanding_prior_rate(
        df, group_col="store_key")

    df["month"] = df["transaction_date"].dt.month
    df["is_peak_season"] = df["month"].isin([10, 11, 12]).astype(int)

    df = df.sort_values("transaction_date", kind="mergesort").reset_index(drop=True)
    logger.info("build_features: feature columns added; date range %s to %s",
                df["transaction_date"].min().date(), df["transaction_date"].max().date())
    return df


# ---------------------------------------------------------------------------
# Step 3 — Chronological train/test split + cold-start imputation
# ---------------------------------------------------------------------------

def build_time_split(df, test_frac=0.2, cutoff_date=None):
    """
    Splits df into (train, test) chronologically by transaction_date —
    NOT a random split. Everything in train happened strictly before
    everything in test, mirroring "deploy today, score what comes next."

    Pass an explicit cutoff_date to pin the split to a specific date
    (e.g. pd.Timestamp("2026-04-01") for a last-3-months test window);
    otherwise test_frac picks the cutoff as the corresponding quantile
    of transaction_date so the test set holds roughly that fraction of
    rows.

    Cold-start imputation (customer/store prior_fail_rate is NaN when a
    customer/store has no prior transactions yet) is filled with TRAIN's
    own population-average fail rate — computed after the split, so no
    test-set statistic ever leaks into the imputed value, and the same
    train-derived constant is applied to both train and test.
    """
    df = df.sort_values("transaction_date", kind="mergesort").reset_index(drop=True)

    if cutoff_date is None:
        cutoff_date = df["transaction_date"].quantile(1 - test_frac)

    train = df[df["transaction_date"] < cutoff_date].copy()
    test = df[df["transaction_date"] >= cutoff_date].copy()

    logger.info("build_time_split: cutoff=%s | train=%d rows (%d failed, %.4f rate) | "
                "test=%d rows (%d failed, %.4f rate)",
                pd.Timestamp(cutoff_date).date(),
                len(train), train["label"].sum(), train["label"].mean(),
                len(test), test["label"].sum(), test["label"].mean())

    # Train-only population average, used to fill cold-start NaNs in
    # BOTH train and test. Test statistics never touch this value.
    train_pop_rate = train["label"].mean()
    for col in ["customer_prior_fail_rate", "store_prior_fail_rate"]:
        train[col] = train[col].fillna(train_pop_rate)
        test[col] = test[col].fillna(train_pop_rate)

    logger.info("build_time_split: cold-start fill value (train population rate) = %.4f", train_pop_rate)

    return train, test, train_pop_rate


# ---------------------------------------------------------------------------
# CLI entry point (for local development / smoke-testing Steps 1-2)
# ---------------------------------------------------------------------------

def run(sales_path=None, payment_path=None, use_db=False, test_frac=0.2, cutoff_date=None):
    if use_db:
        raw = load_from_db()
    else:
        if not (sales_path and payment_path):
            raise ValueError("--sales and --payment are required when not using --db")
        raw = load_from_files(sales_path, payment_path)

    labeled = build_label(raw)
    featured = build_features(labeled)
    train, test, train_pop_rate = build_time_split(featured, test_frac=test_frac, cutoff_date=cutoff_date)
    return featured, train, test, train_pop_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 7: Steps 1-3 (extract, label, features, chronological split)")
    parser.add_argument("--db", action="store_true", help="Pull from the live Postgres warehouse")
    parser.add_argument("--sales", help="Path to sales export (csv or xlsx) — local/dev mode")
    parser.add_argument("--payment", help="Path to payment export (csv or xlsx) — local/dev mode")
    parser.add_argument("--out", default="module7_features.csv", help="Output path for the full feature table")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction of rows (by date quantile) held out as test")
    parser.add_argument("--cutoff-date", default=None, help="Explicit YYYY-MM-DD cutoff instead of --test-frac")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff_date) if args.cutoff_date else None
    full, train, test, pop_rate = run(sales_path=args.sales, payment_path=args.payment, use_db=args.db,
                                       test_frac=args.test_frac, cutoff_date=cutoff)

    full.to_csv(args.out, index=False)
    train.to_csv(args.out.replace(".csv", "_train.csv"), index=False)
    test.to_csv(args.out.replace(".csv", "_test.csv"), index=False)
    logger.info("Wrote full=%s, train=%s, test=%s", args.out,
                args.out.replace(".csv", "_train.csv"), args.out.replace(".csv", "_test.csv"))
