from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(
    title="Product Demand Forecasting API",
    version="1.0"
)

# ===================================
# LOAD MODEL & ENCODERS
# ===================================

model = joblib.load("models/best_model.pkl")
encoders = joblib.load("models/encoders.pkl")


@app.get("/")
def home():
    return {
        "message": "Product Demand Forecasting API is Running!"
    }


@app.post("/predict")
def predict():

    # ===================================
    # LOAD DATA
    # ===================================

    df = pd.read_csv("data/featured_sales_dataset.csv")

    # Create a separate copy for prediction
    predict_df = df.copy()

    # ===================================
    # ENCODE CATEGORICAL FEATURES
    # ===================================

    categorical_cols = [
        "product_name",
        "category",
        "subcategory",
        "store_name",
        "city",
        "state",
        "weekday"
    ]

    for col in categorical_cols:
        if col in predict_df.columns and col in encoders:
            predict_df[col] = encoders[col].transform(
                predict_df[col].astype(str)
            )

    # ===================================
    # FEATURES
    # ===================================

    features = [
        "product_key",
        "store_key",
        "product_name",
        "category",
        "subcategory",
        "store_name",
        "city",
        "state",
        "unit_price",
        "discount",
        "month",
        "quarter",
        "year",
        "weekday",
        "lag_1",
        "lag_3",
        "lag_7",
        "rolling_mean_3",
        "rolling_mean_7"
    ]

    X = predict_df[features]

    # ===================================
    # PREDICTION
    # ===================================

    predictions = model.predict(X)
    predictions = predictions.clip(min=0)

    # ===================================
    # OUTPUT
    # ===================================

    df["predicted_quantity"] = predictions

    output = df[
        [
            "product_key",
            "product_name",
            "store_key",
            "store_name",
            "predicted_quantity"
        ]
    ]

    return output.to_dict(orient="records")