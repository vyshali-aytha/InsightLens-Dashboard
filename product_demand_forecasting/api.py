"""
Product Demand Forecasting — FastAPI service.

Exposes per-product/per-store quantity demand forecasts via REST
endpoints, following the same shape as sales_and_revenue_forecasting's
api.py so the Streamlit dashboard can consume it consistently.

Prefers the precomputed future_demand_forecast.csv (already scored with
predicted_quantity via train_model.py's best model). Falls back to
loading best_model.pkl / encoders.pkl and scoring featured_sales_dataset.csv
live if the precomputed file isn't present.

Run:
    uvicorn api:app --port 8012 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = FastAPI(
    title="Product Demand Forecasting API",
    description="API for per-product/store quantity demand predictions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Paths (resolved relative to this file, not the CWD uvicorn is
# launched from — the original script-style paths broke if run from
# anywhere but this exact folder) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")
FEATURED_DATA_PATH = os.path.join(BASE_DIR, "data", "featured_sales_dataset.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "sales_dataset.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "future_demand_forecast.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.csv")

CATEGORICAL_COLS = ["product_name", "category", "subcategory", "store_name", "city", "state", "weekday"]
FEATURES = [
    "product_key", "store_key", "product_name", "category", "subcategory",
    "store_name", "city", "state", "unit_price", "discount",
    "month", "quarter", "year", "weekday",
    "lag_1", "lag_3", "lag_7", "rolling_mean_3", "rolling_mean_7",
]

model = None
encoders = None
forecast_df = None
model_metrics = {}
last_scored_at = None


def _load_metrics():
    global model_metrics
    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        model_metrics = {row["Model"]: {"mae": row["MAE"], "rmse": row["RMSE"], "wmape": row["WMAPE"]}
                          for _, row in metrics_df.iterrows()}


def load_data_and_model():
    """Load the precomputed forecast if available (fast path). Only
    falls back to loading the model + scoring the featured dataset live
    if future_demand_forecast.csv is missing."""
    global model, encoders, forecast_df, last_scored_at

    _load_metrics()

    if os.path.exists(FORECAST_PATH):
        forecast_df = pd.read_csv(FORECAST_PATH)
        forecast_df["full_date"] = pd.to_datetime(forecast_df["full_date"])

        # future_demand_forecast.csv (written by forecast.py) carries
        # label-encoded product_name/category/subcategory/store_name/
        # city/state instead of the original text — join the real labels
        # back in from the raw extract, keyed on product_key/store_key,
        # so the API (and dashboard) don't show numbers where names
        # should be.
        if os.path.exists(RAW_DATA_PATH) and "product_name" in forecast_df.columns:
            looks_encoded = pd.to_numeric(forecast_df["product_name"], errors="coerce").notna().all()
            if looks_encoded:
                raw = pd.read_csv(RAW_DATA_PATH)
                product_map = raw[["product_key", "product_name", "category", "subcategory", "unit_price"]].drop_duplicates("product_key").set_index("product_key")
                store_map = raw[["store_key", "store_name", "city", "state"]].drop_duplicates("store_key").set_index("store_key")
                for col in ["product_name", "category", "subcategory"]:
                    if col in forecast_df.columns:
                        forecast_df[col] = forecast_df["product_key"].map(product_map[col])
                for col in ["store_name", "city", "state"]:
                    if col in forecast_df.columns:
                        forecast_df[col] = forecast_df["store_key"].map(store_map[col])
    elif os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH) and os.path.exists(FEATURED_DATA_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        df = pd.read_csv(FEATURED_DATA_PATH)
        predict_df = df.copy()
        for col in CATEGORICAL_COLS:
            if col in predict_df.columns and col in encoders:
                predict_df[col] = encoders[col].transform(predict_df[col].astype(str))
        X = predict_df[FEATURES]
        predictions = model.predict(X).clip(min=0)
        df["predicted_quantity"] = predictions
        forecast_df = df
        if "full_date" in forecast_df.columns:
            forecast_df["full_date"] = pd.to_datetime(forecast_df["full_date"])

    last_scored_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


try:
    load_data_and_model()
except Exception as e:
    print(f"Error loading data/model: {e}")


@app.get("/")
def home():
    return {"message": "Product Demand Forecasting API is Running!"}


@app.get("/health")
def health():
    status = "ok" if forecast_df is not None and not forecast_df.empty else "offline"
    return {
        "status": status,
        "model": "product_demand_forecast",
        "rows_scored": len(forecast_df) if forecast_df is not None else 0,
        "last_scored_at": last_scored_at,
        "metrics": model_metrics,
    }


@app.get("/forecast")
def forecast():
    """Full per-transaction quantity forecast — actual vs predicted."""
    if forecast_df is None or forecast_df.empty:
        return []
    df = forecast_df.copy()
    if "full_date" in df.columns:
        df["full_date"] = df["full_date"].astype(str)
    return df.to_dict(orient="records")


@app.get("/forecast/summary")
def forecast_summary(store_key: int | None = None, product_key: int | None = None):
    if forecast_df is None or forecast_df.empty:
        return []
    filtered = forecast_df.copy()
    if store_key is not None and "store_key" in filtered.columns:
        filtered = filtered[filtered["store_key"] == store_key]
    if product_key is not None and "product_key" in filtered.columns:
        filtered = filtered[filtered["product_key"] == product_key]
    if filtered.empty:
        return []
    summary = (
        filtered.groupby(["year", "month"], as_index=False)
        .agg({"quantity": "sum", "predicted_quantity": "sum"})
    )
    return summary.to_dict(orient="records")


@app.post("/predict")
def predict():
    """Legacy endpoint: recompute predictions live from the featured
    dataset (kept for backward compatibility with earlier callers)."""
    if model is None or encoders is None:
        if forecast_df is not None and not forecast_df.empty:
            cols = ["product_key", "product_name", "store_key", "store_name", "predicted_quantity"]
            return forecast_df[[c for c in cols if c in forecast_df.columns]].to_dict(orient="records")
        return {"error": "Model/encoders not loaded and no precomputed forecast available."}

    df = pd.read_csv(FEATURED_DATA_PATH)
    predict_df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in predict_df.columns and col in encoders:
            predict_df[col] = encoders[col].transform(predict_df[col].astype(str))
    X = predict_df[FEATURES]
    predictions = model.predict(X).clip(min=0)
    df["predicted_quantity"] = predictions
    output = df[["product_key", "product_name", "store_key", "store_name", "predicted_quantity"]]
    return output.to_dict(orient="records")
