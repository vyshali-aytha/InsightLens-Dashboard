"""
Sales & Revenue Forecasting — FastAPI service.

Exposes forecast results and historical data via REST endpoints.
Trains a RandomForest model on startup using sales_ml_input.csv,
then serves predictions for the test period (Aug–Dec 2026).

Run:
    uvicorn api:app --port 8011 --reload
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import os

app = FastAPI(
    title="Sales & Revenue Forecasting API",
    description="API for sales revenue predictions using RandomForest.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Data & Model Loading ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_INPUT_PATH = os.path.join(BASE_DIR, "sales_ml_input.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "forecast_results_1.csv")

ml_input_df = None
forecast_df = None
model = None
model_metrics = {}
last_scored_at = None


def load_data_and_train():
    global ml_input_df, forecast_df, model, model_metrics, last_scored_at

    # Load ML input
    ml_input_df = pd.read_csv(ML_INPUT_PATH)
    ml_input_df["full_date"] = pd.to_datetime(ml_input_df["full_date"])
    ml_input_df = ml_input_df.sort_values("full_date")
    ml_input_df = ml_input_df.dropna(subset=["lag_1", "lag_7", "rolling_avg_7"])

    # If precomputed forecast exists, load it directly
    if os.path.exists(FORECAST_PATH):
        forecast_df = pd.read_csv(FORECAST_PATH)
        forecast_df["full_date"] = pd.to_datetime(forecast_df["full_date"])

        # Compute metrics from precomputed results
        mae = mean_absolute_error(forecast_df["Actual Revenue"], forecast_df["Predicted Revenue"])
        rmse = float(np.sqrt(mean_squared_error(forecast_df["Actual Revenue"], forecast_df["Predicted Revenue"])))
        mape = float(np.mean(np.abs(
            (forecast_df["Actual Revenue"] - forecast_df["Predicted Revenue"]) / forecast_df["Actual Revenue"]
        )) * 100)
        model_metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}
    else:
        # Train model from scratch
        features = ["month", "quarter", "year", "weekday", "lag_1", "lag_7", "rolling_avg_7"]
        X = ml_input_df[features]
        y = ml_input_df["total_revenue"]

        train_mask = (ml_input_df["full_date"] >= "2025-01-01") & (ml_input_df["full_date"] <= "2026-07-31")
        test_mask = (ml_input_df["full_date"] >= "2026-08-01") & (ml_input_df["full_date"] <= "2026-12-31")

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_data = ml_input_df.loc[test_mask].copy()
        test_data["Actual Revenue"] = y_test.values
        test_data["Predicted Revenue"] = y_pred
        forecast_df = test_data

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        model_metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}

    last_scored_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Load on startup
try:
    load_data_and_train()
except Exception as e:
    print(f"Error loading data/model: {e}")


@app.get("/health")
def health():
    status = "ok" if forecast_df is not None and not forecast_df.empty else "offline"
    return {
        "status": status,
        "model": "sales_forecast",
        "rows_scored": len(forecast_df) if forecast_df is not None else 0,
        "historical_rows": len(ml_input_df) if ml_input_df is not None else 0,
        "last_scored_at": last_scored_at,
        "metrics": model_metrics,
    }


@app.get("/forecast")
def forecast():
    if forecast_df is None or forecast_df.empty:
        return []
    df = forecast_df.copy()
    df["full_date"] = df["full_date"].astype(str)
    return df.to_dict(orient="records")


@app.get("/historical")
def historical():
    if ml_input_df is None or ml_input_df.empty:
        return []
    df = ml_input_df.copy()
    df["full_date"] = df["full_date"].astype(str)
    return df.to_dict(orient="records")


@app.get("/forecast/summary")
def forecast_summary(
    store_id: int | None = Query(None, description="Filter by store ID"),
    product_id: int | None = Query(None, description="Filter by product ID"),
):
    if forecast_df is None or forecast_df.empty:
        return []
    filtered = forecast_df.copy()
    if store_id is not None:
        filtered = filtered[filtered["store_id"] == store_id]
    if product_id is not None:
        filtered = filtered[filtered["product_id"] == product_id]

    if filtered.empty:
        return []

    summary = (
        filtered
        .groupby(["year", "month"], as_index=False)
        .agg({
            "total_orders": "sum",
            "avg_order_value": "mean",
            "Actual Revenue": "sum",
            "Predicted Revenue": "sum",
        })
    )
    return summary.to_dict(orient="records")
