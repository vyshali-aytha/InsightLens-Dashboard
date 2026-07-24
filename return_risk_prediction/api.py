from pathlib import Path
from typing import Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

app = FastAPI(title="Return Risk Prediction API", version="1.0.0")


class AnalyticsRequest(BaseModel):
    query: str | None = None


def _load_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Required file not found: {name}")
    return pd.read_csv(path)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Return Risk Prediction API is running.",
        "endpoints": [
            "GET /",
            "GET /dashboard",
            "GET /top-returned-products",
            "GET /top-return-cities",
            "GET /return-reasons",
            "GET /high-risk-orders",
            "POST /analytics",
        ],
    }


@app.get("/dashboard")
def dashboard() -> Dict[str, Any]:
    predictions = _load_csv("predictions.csv")
    if predictions.empty:
        raise HTTPException(status_code=404, detail="No prediction data available.")

    total_transactions = int(len(predictions))
    predicted_returns = int((predictions["predicted_return"] == "Yes").sum())
    overall_return_rate = round(float(predicted_returns / total_transactions), 4) if total_transactions else 0.0
    highest_return_city = predictions.groupby("customer_city").size().idxmax() if not predictions.empty else None
    highest_return_product = predictions.groupby("product_key").size().idxmax() if not predictions.empty else None
    highest_return_category = predictions.groupby("category").size().idxmax() if not predictions.empty else None
    top_reason = predictions["normalized_return_reason"].fillna("not returned").value_counts().idxmax() if not predictions.empty else None

    return {
        "total_transactions": total_transactions,
        "predicted_returns": predicted_returns,
        "overall_return_rate": overall_return_rate,
        "highest_return_city": highest_return_city,
        "highest_return_product": highest_return_product,
        "highest_return_category": highest_return_category,
        "top_return_reason": top_reason,
    }


@app.get("/top-returned-products")
def top_returned_products() -> Dict[str, Any]:
    return _load_csv("highly_returned_products.csv").to_dict(orient="records")


@app.get("/top-return-cities")
def top_return_cities() -> Dict[str, Any]:
    return _load_csv("top_return_cities.csv").to_dict(orient="records")


@app.get("/return-reasons")
def return_reasons() -> Dict[str, Any]:
    return _load_csv("return_reason_summary.csv").to_dict(orient="records")


@app.get("/high-risk-orders")
def high_risk_orders() -> Dict[str, Any]:
    predictions = _load_csv("predictions.csv")
    return predictions[predictions["risk_level"].astype(str).str.lower() == "high"].to_dict(orient="records")


@app.post("/analytics")
def analytics(payload: AnalyticsRequest) -> Dict[str, Any]:
    query = (payload.query or "dashboard").strip().lower()
    supported_queries = [
        "dashboard",
        "top_returned_products",
        "top_return_cities",
        "return_reasons",
        "high_risk_orders",
    ]

    if query == "dashboard":
        return dashboard()
    if query == "top_returned_products":
        return top_returned_products()
    if query == "top_return_cities":
        return top_return_cities()
    if query == "return_reasons":
        return return_reasons()
    if query == "high_risk_orders":
        return high_risk_orders()

    return {
        "message": "Invalid query.",
        "supported_queries": supported_queries,
    }
