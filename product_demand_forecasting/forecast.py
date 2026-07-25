import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# =====================================================
# CONFIGURATION
# =====================================================

FORECAST_DAYS = 30

MODEL_PATH = "models/best_model.pkl"
ENCODER_PATH = "models/encoders.pkl"
DATA_PATH = "data/featured_sales_dataset.csv"
OUTPUT_PATH = "forecast_predictions.csv"

# =====================================================
# LOAD MODEL & ENCODERS
# =====================================================

print("=" * 60)
print("Loading Trained Model...")

model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully")

print("Loading Label Encoders...")

encoders = joblib.load(ENCODER_PATH)

print("Encoders Loaded Successfully")

# =====================================================
# LOAD FEATURED DATASET
# =====================================================

print("Loading Featured Dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape :", df.shape)

# =====================================================
# DATE PROCESSING
# =====================================================

df["full_date"] = pd.to_datetime(df["full_date"])

df = df.sort_values(
    ["product_key", "store_key", "full_date"]
)

print("Date Conversion Completed")

# =====================================================
# GET LATEST RECORD OF EVERY PRODUCT-STORE
# =====================================================

latest_records = (
    df.groupby(
        ["product_key", "store_key"],
        as_index=False
    )
    .tail(1)
    .reset_index(drop=True)
)

print("Unique Product-Store Pairs :", len(latest_records))

# =====================================================
# PREPARE FORECAST CONTAINER
# =====================================================

forecast_rows = []

print("=" * 60)
print("Preparing Forecast Data...")
print("=" * 60)

# =====================================================
# RECURSIVE FORECASTING
# =====================================================

print("Starting Recursive Forecasting...")

for _, row in latest_records.iterrows():

    history = [
        row["lag_7"],
        row["lag_3"],
        row["lag_1"]
    ]

    current_date = row["full_date"]

    for day in range(1, FORECAST_DAYS + 1):

        future_date = current_date + timedelta(days=day)

        feature_row = {
            "product_key": row["product_key"],
            "store_key": row["store_key"],
            "product_name": row["product_name"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "store_name": row["store_name"],
            "city": row["city"],
            "state": row["state"],
            "unit_price": row["unit_price"],
            "discount": row["discount"],

            "month": future_date.month,
            "quarter": future_date.quarter,
            "year": future_date.year,
            "weekday": future_date.weekday(),

            "lag_1": history[-1],
            "lag_3": np.mean(history[-3:]),
            "lag_7": np.mean(history),

            "rolling_mean_3": np.mean(history[-3:]),
            "rolling_mean_7": np.mean(history)
        }

        X = pd.DataFrame([feature_row])

        prediction = model.predict(X)[0]

        prediction = max(0, round(float(prediction), 2))

        forecast_rows.append({
            "forecast_date": future_date,
            "product_key": row["product_key"],
            "product_name": row["product_name"],
            "store_key": row["store_key"],
            "store_name": row["store_name"],
            "city": row["city"],
            "state": row["state"],
            "predicted_quantity": prediction
        })

        history.append(prediction)

        if len(history) > 7:
            history.pop(0)

print("Forecast Generation Completed")