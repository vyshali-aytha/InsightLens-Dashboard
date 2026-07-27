"""
Shared data access for every model page.

Every model in config.MODELS is expected to expose the same shape of
service (this mirrors module7_api.py): a /health check, a /scores
endpoint, and one or more /charts/<name> endpoints. This module tries
the live API first and falls back to local CSVs in <csv_dir> so the
dashboard still works when a model's API isn't running (e.g. during
early development, or on a laptop with no server started).

Nothing here is model-specific — the same three functions serve every
model. A new model works automatically once it follows the same API
shape and has its config entry in config.py.
"""

import os
import requests
import pandas as pd
import streamlit as st

REQUEST_TIMEOUT = 3  # seconds — dashboard should fail fast to its CSV fallback, not hang


def _api_get(base_url: str, path: str):
    if not base_url:
        return None
    try:
        resp = requests.get(f"{base_url}{path}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _csv_fallback(csv_dir: str, filename: str) -> pd.DataFrame | None:
    path = os.path.join(csv_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=60, show_spinner=False)
def get_health(model_key: str, api_base: str) -> dict:
    """Returns {'status': 'ok' | 'degraded' | 'offline', ...}"""
    result = _api_get(api_base, "/health")
    if result is not None:
        return result
    return {"status": "offline", "model": model_key, "last_error": "API unreachable — showing cached CSVs if available"}


@st.cache_data(ttl=60, show_spinner=False)
def get_scores(model_key: str, model_cfg: dict, flagged_only: bool = False, limit: int | None = None) -> pd.DataFrame | None:
    """Per-transaction / per-record scores table."""
    params = {}
    if flagged_only:
        params["flagged_only"] = "true"
    if limit:
        params["limit"] = limit

    api_base = model_cfg["api_base"]
    if api_base:
        try:
            resp = requests.get(f"{api_base}{model_cfg['endpoints']['scores']}", params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json())
            if not df.empty:
                return df
        except requests.RequestException:
            pass

    # Fallback: look for a *_scores.csv in the model's local data folder
    df = _csv_fallback(model_cfg["csv_dir"], f"{model_key}_scores.csv")
    if df is not None:
        if flagged_only and "flagged" in df.columns:
            df = df[df["flagged"]]
        if limit:
            df = df.head(limit)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def get_chart_data(model_key: str, model_cfg: dict, chart_key: str) -> pd.DataFrame | None:
    """Chart-ready aggregate table for one of the model's declared charts."""
    api_base = model_cfg["api_base"]
    chart_path = model_cfg["endpoints"].get("charts", {}).get(chart_key)
    if api_base and chart_path:
        result = _api_get(api_base, chart_path)
        if result is not None:
            df = pd.DataFrame(result)
            if not df.empty:
                return df

    return _csv_fallback(model_cfg["csv_dir"], f"{model_key}_scores_{chart_key}.csv")


# ---------------------------------------------------------------------------
# type: return_dashboard (Return Risk Prediction's api.py shape)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_return_dashboard(model_cfg: dict) -> dict | None:
    """Summary KPIs from /dashboard, or computed from predictions.csv if offline."""
    api_base = model_cfg["api_base"]
    result = _api_get(api_base, model_cfg["endpoints"]["dashboard"])
    if result is not None:
        return result

    df = _csv_fallback(model_cfg["csv_dir"], model_cfg["csv_files"]["predictions"])
    if df is None or df.empty:
        return None
    total = len(df)
    predicted_returns = int((df["predicted_return"] == "Yes").sum())
    return {
        "total_transactions": total,
        "predicted_returns": predicted_returns,
        "overall_return_rate": round(predicted_returns / total, 4) if total else 0.0,
        "highest_return_city": df.groupby("customer_city").size().idxmax(),
        "highest_return_product": df.groupby("product_key").size().idxmax(),
        "highest_return_category": df.groupby("category").size().idxmax(),
        "top_return_reason": df["normalized_return_reason"].fillna("not returned").value_counts().idxmax(),
    }


@st.cache_data(ttl=60, show_spinner=False)
def get_return_predictions(model_cfg: dict) -> pd.DataFrame | None:
    """Transaction-level Return Risk predictions used for client-side analysis.

    The Return Risk service exposes aggregates rather than a full prediction
    endpoint.  Its existing predictions.csv output is therefore the canonical
    source for interactive filtering when the dashboard is running locally.
    """
    return _csv_fallback(model_cfg["csv_dir"], model_cfg["csv_files"]["predictions"])


@st.cache_data(ttl=60, show_spinner=False)
def get_return_breakdown(model_cfg: dict, breakdown_key: str) -> pd.DataFrame | None:
    """One of: top_returned_products, top_return_cities, return_reasons, high_risk_orders."""
    api_base = model_cfg["api_base"]
    if breakdown_key == "high_risk_orders":
        result = _api_get(api_base, model_cfg["endpoints"]["high_risk_orders"])
        if result is not None:
            return pd.DataFrame(result)
        df = _csv_fallback(model_cfg["csv_dir"], model_cfg["csv_files"]["predictions"])
        if df is not None:
            return df[df["risk_level"].astype(str).str.lower() == "high"]
        return None

    endpoint = model_cfg["endpoints"].get(breakdown_key)
    if api_base and endpoint:
        result = _api_get(api_base, endpoint)
        if result is not None:
            return pd.DataFrame(result)

    csv_name = model_cfg["csv_files"].get(breakdown_key)
    if csv_name:
        return _csv_fallback(model_cfg["csv_dir"], csv_name)
    return None


# ---------------------------------------------------------------------------
# type: discount_calculator (Discount Impact's api.py shape)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_discount_products(model_cfg: dict) -> pd.DataFrame | None:
    """Product picklist — always from the local dim_product.csv; the API
    itself has no endpoint to list products, only to price one."""
    df = _csv_fallback(model_cfg["csv_dir"], model_cfg["products_csv"])
    if df is not None:
        df = df[["product_id", "product_name", "category", "unit_price"]].drop_duplicates().reset_index(drop=True)
    return df


def post_discount_prediction(model_cfg: dict, product_id: str, discount: float) -> dict | None:
    """Calls POST /predict_discount. No CSV fallback is possible here —
    this is a live what-if calculation, not a precomputed batch result."""
    api_base = model_cfg["api_base"]
    if not api_base:
        return None
    try:
        resp = requests.post(
            f"{api_base}{model_cfg['endpoints']['predict']}",
            json={"product_id": product_id, "discount": discount},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return None
    except requests.RequestException as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# type: sales_forecast (Sales & Revenue Forecasting)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_sales_forecast_data(model_cfg: dict) -> pd.DataFrame | None:
    """Forecast results — actual vs predicted revenue for the test period.

    Tries the live API first (/forecast), falls back to forecast_results_1.csv.
    """
    api_base = model_cfg.get("api_base")
    endpoint = model_cfg.get("endpoints", {}).get("forecast")
    if api_base and endpoint:
        result = _api_get(api_base, endpoint)
        if result is not None:
            df = pd.DataFrame(result)
            if not df.empty:
                return df
    return _csv_fallback(model_cfg["csv_dir"], model_cfg["csv_files"]["forecast"])


@st.cache_data(ttl=60, show_spinner=False)
def get_sales_historical_data(model_cfg: dict) -> pd.DataFrame | None:
    """Full historical ML input — training + test period actuals.

    Tries the live API first (/historical), falls back to sales_ml_input.csv.
    """
    api_base = model_cfg.get("api_base")
    endpoint = model_cfg.get("endpoints", {}).get("historical")
    if api_base and endpoint:
        result = _api_get(api_base, endpoint)
        if result is not None:
            df = pd.DataFrame(result)
            if not df.empty:
                return df
    return _csv_fallback(model_cfg["csv_dir"], model_cfg["csv_files"]["historical"])


# ---------------------------------------------------------------------------
# type: demand_forecast (Product Demand Forecasting)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_demand_forecast_data(model_cfg: dict) -> pd.DataFrame | None:
    """Per-transaction quantity forecast — actual vs predicted, per
    product/store. Tries the live API first (/forecast), falls back to
    future_demand_forecast.csv."""
    api_base = model_cfg.get("api_base")
    endpoint = model_cfg.get("endpoints", {}).get("forecast")
    if api_base and endpoint:
        result = _api_get(api_base, endpoint)
        if result is not None:
            df = pd.DataFrame(result)
            if not df.empty:
                return df
    return _csv_fallback(model_cfg["csv_dir"], model_cfg["csv_files"]["forecast"])


@st.cache_data(ttl=300, show_spinner=False)
def get_demand_model_metrics(model_cfg: dict) -> pd.DataFrame | None:
    """Model comparison table (MAE/RMSE/WMAPE per candidate model) from
    model_metrics.csv — same for API-backed or CSV-fallback mode."""
    return _csv_fallback(model_cfg["csv_dir"], model_cfg["csv_files"]["metrics"])


# ---------------------------------------------------------------------------
# type: pipeline_dashboard (Flask API's project/app.py shape — POST /run,
# then cached GET results/csv/graphs). No CSV fallback: the Flask API is
# the only data source (Dashboard -> HTTP -> Flask API -> Warehouse).
# ---------------------------------------------------------------------------
PIPELINE_RUN_TIMEOUT = 300  # seconds — pipelines hit the DB and can train ML models


def post_pipeline_run(model_cfg: dict) -> dict | None:
    """POST the model's /run endpoint. Executes the pipeline — call only on
    explicit user action, never automatically."""
    api_base = model_cfg["api_base"]
    endpoint = model_cfg["endpoints"]["run"]
    if not api_base:
        return None
    try:
        resp = requests.post(f"{api_base}{endpoint}", timeout=PIPELINE_RUN_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        # app.py's error routes return {"error": "..."} in the body even on
        # a 500 — raise_for_status() already fired, so read the body
        # ourselves instead of losing it behind a generic "500 Server
        # Error" string.
        detail = None
        try:
            detail = exc.response.json().get("error")
        except (ValueError, AttributeError):
            pass
        return {"error": detail or str(exc)}
    except requests.RequestException as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=60, show_spinner=False)
def get_pipeline_json(model_key: str, model_cfg: dict, endpoint_key: str):
    """Cached GET against one of the model's declared endpoints. The Flask
    API already serves these from its in-memory cache (populated by /run),
    so this never triggers recomputation — it only avoids redundant HTTP
    round-trips from Streamlit's own reruns."""
    api_base = model_cfg["api_base"]
    path = model_cfg["endpoints"].get(endpoint_key)
    return _api_get(api_base, path)


def pipeline_graph_url(model_cfg: dict, graph_name: str) -> str | None:
    """Direct URL to a cached graph PNG on the Flask API — passed straight
    to st.image so the browser fetches it, instead of Streamlit re-reading
    and re-serving the bytes itself."""
    api_base = model_cfg["api_base"]
    if not api_base:
        return None
    return f"{api_base}{model_cfg['endpoints']['graph_file'].format(name=graph_name)}"


@st.cache_data(ttl=300, show_spinner=False)
def get_pipeline_csv_bytes(model_cfg: dict, filename: str) -> bytes | None:
    """Fetches one generated CSV's bytes for a download button. Cached so
    re-rendering the download-buttons tab doesn't re-fetch every file on
    every rerun — cache clears naturally alongside get_pipeline_json when
    a new pipeline run happens (both keyed off server state that only
    changes via POST /run)."""
    api_base = model_cfg["api_base"]
    if not api_base:
        return None
    try:
        resp = requests.get(
            f"{api_base}{model_cfg['endpoints']['csv_file'].format(filename=filename)}",
            timeout=REQUEST_TIMEOUT * 4,
        )
        resp.raise_for_status()
        return resp.content
    except requests.RequestException:
        return None

