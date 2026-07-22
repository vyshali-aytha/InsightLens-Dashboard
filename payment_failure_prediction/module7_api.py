"""
module7_api.py
---------------
Module 7: Payment Failure Risk Scoring — exposed as an API.

This wraps the existing pipeline (module7_features.py + module7_score.py)
in a small FastAPI web server. Nothing about the model or the feature
logic changes — this file only adds an HTTP layer on top of functions
that already existed, so a dashboard (or a teammate's script, or curl)
can ask for this module's output over a URL instead of needing direct
access to this codebase's files or database.

Run it:
    pip install fastapi uvicorn
    uvicorn module7_api:app --host 0.0.0.0 --port 8007 --reload

Then, with the server running, these URLs are live in a browser or via
curl / requests / a dashboard's HTTP client:

    GET  http://localhost:8007/health
        -> {"status": "ok", "model": "module7_model.joblib", "rows_scored": 12000}

    GET  http://localhost:8007/scores
        -> per-transaction risk scores (the main output)

    GET  http://localhost:8007/scores?flagged_only=true&limit=50
        -> just the riskiest 50 flagged transactions

    GET  http://localhost:8007/charts/by_method_provider
        -> chart-ready: risk aggregated by payment_method x payment_provider
        -> shape: [{"payment_method": "...", "payment_provider": "...",
                    "n_transactions": N, "mean_risk_score": 0.xx,
                    "actual_fail_rate": 0.xx, ...}, ...]

    GET  http://localhost:8007/charts/by_month
        -> chart-ready: risk aggregated by calendar month (seasonality)
        -> same shape idea, grouped by "month" instead

    POST http://localhost:8007/refresh
        -> re-runs scoring against the live warehouse right now, instead
           of waiting for the next scheduled refresh (see _get_scored below)

Every endpoint returns plain JSON (lists of objects / key-value pairs) —
no special client library needed to consume this from any language.
"""

import logging
import os
import threading
import time
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from module7_features import build_features, load_from_db, load_from_files
from module7_score import score, summarize_by_method_provider, summarize_by_month, CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger("module7_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODEL_PATH = os.environ.get("MODULE7_MODEL_PATH", "module7_model.joblib")
CAPACITY_PCT = float(os.environ.get("MODULE7_CAPACITY_PCT", "10.0"))
# How often the background refresh loop re-scores against the live
# warehouse, in seconds. Keeps /scores fast (served from memory) while
# still staying reasonably current without a request having to wait on
# a full feature-build + scoring pass every single time.
REFRESH_INTERVAL_SECONDS = int(os.environ.get("MODULE7_REFRESH_SECONDS", "300"))
# Local file fallback, for running this without a live DB connection —
# same idea as module7_score.py's --sales/--payment flags.
LOCAL_SALES_PATH = os.environ.get("MODULE7_SALES_PATH")
LOCAL_PAYMENT_PATH = os.environ.get("MODULE7_PAYMENT_PATH")

app = FastAPI(
    title="InsightLens — Module 7: Payment Failure Risk API",
    description="Payment risk scores and dashboard-ready aggregates for the Streamlit dashboard.",
    version="1.0",
)

# CORS: allows a dashboard running on a different host/port (e.g. the
# Streamlit app on :8501) to call this API's URLs directly from the
# browser. Locked down to be permissive here since this runs on an
# internal network; tighten allow_origins to specific hosts before this
# is ever exposed outside a trusted network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_state_lock = threading.Lock()
_state = {
    "scored_df": None,       # cached DataFrame from the last scoring run
    "last_scored_at": None,  # when that run happened
    "rows_scored": 0,
    "error": None,           # last refresh error, if any (so /health can report it)
}


def _load_raw() -> pd.DataFrame:
    if LOCAL_SALES_PATH and LOCAL_PAYMENT_PATH:
        return load_from_files(LOCAL_SALES_PATH, LOCAL_PAYMENT_PATH)
    return load_from_db()


def _run_scoring() -> pd.DataFrame:
    """One full pass: load current data, build features, score. This is
    the exact same logic module7_score.py's `run()` uses — nothing new is
    invented here, it's reused so the API can never drift from the CLI
    tool's behavior."""
    raw = _load_raw()
    raw["label"] = (raw["payment_status"] == "FAILED").astype(int)  # only used for cold-start pop_rate fallback
    featured = build_features(raw)
    scored = score(_model, featured)

    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    scored["rank_pct"] = (scored.index + 1) / len(scored)
    scored["flagged"] = scored["rank_pct"] <= (CAPACITY_PCT / 100.0)
    return scored


def _refresh_once():
    try:
        scored = _run_scoring()
        with _state_lock:
            _state["scored_df"] = scored
            _state["last_scored_at"] = pd.Timestamp.now()
            _state["rows_scored"] = len(scored)
            _state["error"] = None
        logger.info("refresh: scored %d payment(s)", len(scored))
    except Exception as exc:  # noqa: BLE001 — surface any failure via /health rather than crashing the loop
        with _state_lock:
            _state["error"] = str(exc)
        logger.exception("refresh: scoring run failed")


def _refresh_loop():
    # startup() already ran one synchronous _refresh_once() before this
    # thread starts, so sleep first here — otherwise every server start
    # triggers two back-to-back scoring passes (a wasted DB round-trip,
    # and confusing to anyone watching the logs expecting one refresh).
    while True:
        time.sleep(REFRESH_INTERVAL_SECONDS)
        _refresh_once()


def _assert_model_matches_current_features(model):
    """
    Guards against exactly the failure mode found in an earlier deployment:
    a module7_model.joblib trained by an OLDER module7_train.py (missing
    is_peak_season, no calibration) got loaded and served silently by
    CURRENT module7_score.py/module7_api.py code, which expected it —
    scores looked plausible (numbers in [0,1]) but were badly miscalibrated
    (mean risk_score ~0.41 vs. an actual ~4% base rate) because the model
    never saw the newer feature at all. sklearn happily loads any old
    pickled Pipeline and happily scores extra unused DataFrame columns
    without erroring, so nothing caught this automatically before — this
    check makes that fail LOUDLY at startup instead of silently in
    production.
    """
    try:
        prep = model.named_steps["prep"] if hasattr(model, "named_steps") else None
        if prep is None and hasattr(model, "calibrated_classifiers_"):
            cc = model.calibrated_classifiers_[0]
            pipeline = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            prep = pipeline.named_steps["prep"] if pipeline is not None else None
        if prep is None:
            logger.warning("startup: could not introspect model to verify its feature set — "
                            "skipping the mismatch check, proceeding anyway")
            return
        cat_cols = None
        for name, transformer, cols in prep.transformers_:
            if name == "cat":
                cat_cols = list(cols)
                break
        model_categorical = cat_cols
    except Exception as exc:  # noqa: BLE001 — this is a best-effort safety net, not core logic
        logger.warning("startup: feature-mismatch check failed to run (%s) — proceeding anyway", exc)
        return

    expected = list(CATEGORICAL_FEATURES)
    if set(model_categorical) != set(expected):
        raise RuntimeError(
            f"module7_model.joblib was trained on categorical features {model_categorical}, "
            f"but the current code expects {expected}. This model is STALE relative to the "
            f"current module7_train.py/module7_score.py — retrain in the main pipeline project "
            f"(python module7_train.py --train ... --test ... --model-out module7_model.joblib) "
            f"and redeploy the new .joblib file before starting this API."
        )
    logger.info("startup: model feature check passed — categorical features match current code")


@app.on_event("startup")
def startup():
    global _model
    logger.info("startup: loading model from %s", MODEL_PATH)
    _model = joblib.load(MODEL_PATH)
    _assert_model_matches_current_features(_model)
    # First scoring pass happens synchronously so /scores has data the
    # moment the server reports itself as up, instead of a client racing
    # a background thread on first request.
    _refresh_once()
    # Subsequent passes run on a background timer so the API stays fast
    # (served from an in-memory cache) without every request paying the
    # cost of a full feature-build + scoring pass.
    threading.Thread(target=_refresh_loop, daemon=True).start()


def _get_scored() -> pd.DataFrame:
    with _state_lock:
        df = _state["scored_df"]
        error = _state["error"]
    if df is None:
        raise HTTPException(status_code=503, detail=f"No scored data yet. Last error: {error}")
    return df


@app.get("/health")
def health():
    """Lets a dashboard show 'model service down' instead of a raw
    connection error — check this before calling the data endpoints."""
    with _state_lock:
        ok = _state["scored_df"] is not None and _state["error"] is None
        return {
            "status": "ok" if ok else "degraded",
            "model": MODEL_PATH,
            "rows_scored": _state["rows_scored"],
            "last_scored_at": _state["last_scored_at"].isoformat() if _state["last_scored_at"] else None,
            "last_error": _state["error"],
        }


@app.get("/scores")
def scores(
    flagged_only: bool = Query(False, description="Only return payments flagged for review"),
    limit: Optional[int] = Query(None, description="Max number of rows to return, highest risk first"),
):
    """Per-transaction risk scores — the main output of this module."""
    df = _get_scored()
    out_cols = ["payment_id", "transaction_id", "risk_score", "rank_pct", "flagged", "month", "is_peak_season"]
    result = df[out_cols]
    if flagged_only:
        result = result[result["flagged"]]
    if limit:
        result = result.head(limit)
    return result.to_dict(orient="records")


@app.get("/charts/by_method_provider")
def chart_by_method_provider():
    """Chart-ready: risk aggregated by payment_method x payment_provider —
    bar chart of mean_risk_score / actual_fail_rate per combo."""
    df = _get_scored()
    summary = summarize_by_method_provider(df)
    return summary.to_dict(orient="records")


@app.get("/charts/by_month")
def chart_by_month():
    """Chart-ready: risk aggregated by calendar month — line/bar chart
    for the seasonality pattern (Oct/Nov/Dec elevated failure rate)."""
    df = _get_scored()
    summary = summarize_by_month(df)
    return summary.to_dict(orient="records")


@app.post("/refresh")
def refresh_now():
    """Triggers an immediate re-score instead of waiting for the next
    scheduled background refresh — useful right after new payments have
    landed and you don't want to wait REFRESH_INTERVAL_SECONDS."""
    _refresh_once()
    with _state_lock:
        if _state["error"]:
            raise HTTPException(status_code=500, detail=_state["error"])
        return {"status": "refreshed", "rows_scored": _state["rows_scored"]}
