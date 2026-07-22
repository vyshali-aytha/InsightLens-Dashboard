"""
generate_payment_labels.py
---------------------------
Regenerates `payment_status` in payment.csv so it actually depends on
other columns, instead of being (as verified) essentially independent
noise in the current synthetic dataset.

Injected signal, chosen to mirror plausible real-world failure drivers:
  1. ~5% of customers are flagged "high-risk" (e.g. worn-out cards,
     shaky bank accounts) -> substantially higher failure rate on
     every one of their transactions.
  2. ~4% of stores are flagged "high-risk" (e.g. flaky POS/terminal
     integration at that location) -> higher failure rate for
     transactions at that store, regardless of customer.
  3. Certain payment_providers (wallet apps + BHIM) are flagged
     lower-reliability -> a flat bump in failure rate.
  4. Larger transaction amounts get progressively more likely to fail
     (banks/issuers often decline unusually large charges pending
     verification).
  5. Oct/Nov/Dec (holiday/peak shopping season) get a moderate flat
     failure-rate bump — higher volume strains payment infra, more
     first-time/occasional customers transact, and processors see more
     network congestion during this window. Added after a reviewer
     check found monthly volume correlates with monthly failure rate
     in the original data, but the effect wasn't actually DESIGNED in
     (this script previously read sales.csv but never used it) — this
     version merges in the real transaction date and adds a proper,
     deliberate seasonal term so the pattern is consistent and
     statistically real across resamples, not an accidental artifact.

Everything else in payment.csv (payment_id, transaction_id, store
key, customer key, payment_method, payment_provider, amount,
timestamps, is_active) is left untouched — only payment_status is
resampled. Works with either schema: the original int-keyed
store_key/customer_key, or the S001/C001-style store_id/customer_id
used in the MART-format export — whichever columns are present.

Usage:
    python generate_payment_labels.py --sales sales.csv --payment payment.csv --out payment.csv
"""

import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SEED = 42
RISKY_PROVIDERS = {"Mobikwik", "Paytm Wallet", "Amazon Pay Wallet", "BHIM"}
PEAK_MONTHS = {10, 11, 12}

BASE_FAIL_LOGIT = -4.35       # baseline failure rate before any risk bumps (slightly
                               # lower than before to offset the new seasonal term,
                               # so the overall rate stays in a realistic ~4% range)
CUSTOMER_RISK_FRAC = 0.05     # 5% of customers flagged high-risk
CUSTOMER_RISK_BONUS = 3.0     # logit bump for a high-risk customer's transactions
STORE_RISK_FRAC = 0.04        # 4% of stores flagged high-risk
STORE_RISK_BONUS = 2.2        # logit bump for a high-risk store's transactions
PROVIDER_RISK_BONUS = 1.6     # logit bump for a low-reliability provider
AMOUNT_ZSCORE_COEF = 0.55     # logit bump per standard deviation of log(amount)
PEAK_SEASON_BONUS = 0.45      # logit bump for Oct/Nov/Dec transactions — roughly a
                               # 1.5x odds multiplier, deliberately in the same range
                               # as the amount effect: a real, moderate driver, not as
                               # strong as a flagged risky customer/store/provider

PENDING_RATE = 0.01           # flat, independent of risk (transient/system issue, not risk-driven)


def _read(path):
    with open(path, "rb") as f:
        magic = f.read(4)
    return pd.read_excel(path) if magic[:2] == b"PK" else pd.read_csv(path)


def _key_col(df, base_name):
    """Returns whichever of f'{base_name}_key' / f'{base_name}_id' is present."""
    for candidate in (f"{base_name}_key", f"{base_name}_id"):
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Neither {base_name}_key nor {base_name}_id found in columns: {list(df.columns)}")


def _parse_sales_datetime(sales):
    """Handles either a date_key (YYYYMMDD int) or transaction_date_time
    (dd-mm-YYYY HH:MM string) column, whichever the sales export has."""
    if "date_key" in sales.columns:
        return pd.to_datetime(sales["date_key"].astype(int).astype(str), format="%Y%m%d")
    if "transaction_date_time" in sales.columns:
        return pd.to_datetime(sales["transaction_date_time"], format="%d-%m-%Y %H:%M")
    raise KeyError(f"No recognized date column in sales columns: {list(sales.columns)}")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(sales_path, payment_path):
    rng = np.random.default_rng(SEED)

    sales = _read(sales_path)
    pay = _read(payment_path)

    cust_col = _key_col(pay, "customer")
    store_col = _key_col(pay, "store")

    # --- pick the risky cohorts, once, reproducibly ---
    all_customers = np.sort(pay[cust_col].unique())
    n_risky_cust = max(1, int(len(all_customers) * CUSTOMER_RISK_FRAC))
    risky_customers = set(rng.choice(all_customers, size=n_risky_cust, replace=False))

    all_stores = np.sort(pay[store_col].unique())
    n_risky_store = max(1, int(len(all_stores) * STORE_RISK_FRAC))
    risky_stores = set(rng.choice(all_stores, size=n_risky_store, replace=False))

    logger.info("Flagged %d/%d customers and %d/%d stores as high-risk (seed=%d)",
                len(risky_customers), len(all_customers), len(risky_stores), len(all_stores), SEED)

    # --- bring in real transaction dates for the seasonal term ---
    sales_dt = sales[["transaction_id"]].copy()
    sales_dt["transaction_date"] = _parse_sales_datetime(sales)
    df = pay.merge(sales_dt, on="transaction_id", how="left")
    n_unmatched = df["transaction_date"].isna().sum()
    if n_unmatched:
        logger.warning("%d payment row(s) had no matching sales date — treated as non-peak", n_unmatched)
    is_peak = df["transaction_date"].dt.month.isin(PEAK_MONTHS).fillna(False)

    # --- build per-row failure logit ---
    log_amount = np.log1p(df["amount"].astype(float))
    amount_z = (log_amount - log_amount.mean()) / log_amount.std()

    logit = np.full(len(df), BASE_FAIL_LOGIT, dtype=float)
    logit += df[cust_col].isin(risky_customers).to_numpy() * CUSTOMER_RISK_BONUS
    logit += df[store_col].isin(risky_stores).to_numpy() * STORE_RISK_BONUS
    logit += df["payment_provider"].isin(RISKY_PROVIDERS).to_numpy() * PROVIDER_RISK_BONUS
    logit += amount_z.to_numpy() * AMOUNT_ZSCORE_COEF
    logit += is_peak.to_numpy() * PEAK_SEASON_BONUS

    fail_prob = sigmoid(logit)

    # --- sample outcomes: FAILED / PENDING / SUCCESS ---
    draw = rng.random(len(df))
    status = np.full(len(df), "SUCCESS", dtype=object)
    status[draw < fail_prob] = "FAILED"
    # PENDING drawn independently from the leftover SUCCESS pool
    still_success = status == "SUCCESS"
    pending_draw = rng.random(len(df))
    status[still_success & (pending_draw < PENDING_RATE)] = "PENDING"

    # payment_status is the only column that changes — drop the join-only
    # transaction_date column so output schema exactly matches the input
    df["payment_status"] = status
    df = df.drop(columns=["transaction_date"])

    overall = pd.Series(status).value_counts(normalize=True)
    logger.info("New payment_status distribution:\n%s", overall.to_string())

    # sanity: risky cohorts and peak season should show a clearly higher
    # fail rate than baseline
    is_failed = (status == "FAILED").astype(int)
    logger.info("Fail rate — risky customers: %.4f | other customers: %.4f",
                is_failed[df[cust_col].isin(risky_customers)].mean(),
                is_failed[~df[cust_col].isin(risky_customers)].mean())
    logger.info("Fail rate — risky stores: %.4f | other stores: %.4f",
                is_failed[df[store_col].isin(risky_stores)].mean(),
                is_failed[~df[store_col].isin(risky_stores)].mean())
    logger.info("Fail rate — risky providers: %.4f | other providers: %.4f",
                is_failed[df["payment_provider"].isin(RISKY_PROVIDERS)].mean(),
                is_failed[~df["payment_provider"].isin(RISKY_PROVIDERS)].mean())
    logger.info("Fail rate — peak season (Oct/Nov/Dec): %.4f | rest of year: %.4f",
                is_failed[is_peak.to_numpy()].mean(),
                is_failed[~is_peak.to_numpy()].mean())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate payment_status with a real, learnable signal")
    parser.add_argument("--sales", required=True)
    parser.add_argument("--payment", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = generate(args.sales, args.payment)
    result.to_csv(args.out, index=False)
    logger.info("Wrote %d rows to %s", len(result), args.out)
