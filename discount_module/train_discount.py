import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

df = pd.read_csv("sales_new.csv")

print("=" * 70)
print("Discount Impact Model")
print("=" * 70)

print(f"\nDataset Shape : {df.shape}")

# ---------------------------------------------------------
# Remove duplicates
# ---------------------------------------------------------

df = df.drop_duplicates()

# ---------------------------------------------------------
# Remove missing values
# ---------------------------------------------------------

df = df.dropna()

print(f"Dataset After Cleaning : {df.shape}")

# ---------------------------------------------------------
# Feature Columns
# ---------------------------------------------------------

feature_columns = [

    "product_id",
    "category",
    "unit_price",
    "discount"

]

X = df[feature_columns]

y = df["quantity"]

# ---------------------------------------------------------
# Train Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)

# ---------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------

categorical_features = [

    "product_id",
    "category"

]

numeric_features = [

    "unit_price",
    "discount"

]

preprocessor = ColumnTransformer(

    transformers=[

        (

            "cat",

            OneHotEncoder(handle_unknown="ignore"),

            categorical_features

        ),

        (

            "num",

            "passthrough",

            numeric_features

        )

    ]

)

# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

rf = RandomForestRegressor(

    n_estimators=500,

    max_depth=20,

    random_state=42,

    n_jobs=1

)

model = Pipeline([

    ("preprocessor", preprocessor),

    ("model", rf)

])

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\nTraining Model...\n")

model.fit(

    X_train,

    y_train

)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

prediction = model.predict(X_test)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

r2 = r2_score(

    y_test,

    prediction

)

mae = mean_absolute_error(

    y_test,

    prediction

)

rmse = np.sqrt(

    mean_squared_error(

        y_test,

        prediction

    )

)

print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"R² Score              : {r2:.4f}")

print(f"Mean Absolute Error   : {mae:.2f}")

print(f"Root Mean Sq Error    : {rmse:.2f}")

print("=" * 70)

# ---------------------------------------------------------
# Save everything
# ---------------------------------------------------------

joblib.dump(

    {

        "model": model,

        "feature_columns": feature_columns,

        "metrics": {

            "r2": r2,

            "mae": mae,

            "rmse": rmse

        }

    },

    "discount_model.pkl"

)

print("\nModel Saved Successfully")

print("\ndiscount_model.pkl created")