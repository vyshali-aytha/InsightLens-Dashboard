import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ---------------------------------------
# Database Connection
# ---------------------------------------
conn = psycopg2.connect(
    host="localhost",
    database="bbi",
    user="postgres",
    password="Channa@14"
)

# ---------------------------------------
# Load Data from Forecast View
# ---------------------------------------
query = "SELECT * FROM public.sales_ml_input"
df = pd.read_sql(query, conn)

df = pd.read_sql(query, conn)

print(df.columns.tolist())
print(df.head())

# ---------------------------------------
# Data Preparation
# ---------------------------------------

# Convert full_date to datetime
df["full_date"] = pd.to_datetime(df["full_date"])

# Sort by date
df = df.sort_values("full_date")

# Remove rows having NULL lag values
df = df.dropna(subset=["lag_1", "lag_7", "rolling_avg_7"])

# ---------------------------------------
# Prepare Features & Target
# ---------------------------------------

X = df[[
    "month",
    "quarter",
    "year",
    "weekday",
    "lag_1",
    "lag_7",
    "rolling_avg_7"
]]

y = df["total_revenue"]

# ---------------------------------------
# Train/Test Split
# ---------------------------------------

train_mask = (
    (df["full_date"] >= "2025-01-01") &
    (df["full_date"] <= "2026-07-31")
)

test_mask = (
    (df["full_date"] >= "2026-08-01") &
    (df["full_date"] <= "2026-12-31")
)

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("Training rows :", len(X_train))
print("Testing rows  :", len(X_test))

print("Training period:",
      df.loc[train_mask, "full_date"].min(),
      "to",
      df.loc[train_mask, "full_date"].max())

print("Testing period:",
      df.loc[test_mask, "full_date"].min(),
      "to",
      df.loc[test_mask, "full_date"].max())


# print("=" * 50)
# print("Total rows:", len(df))

# print("Min date:", df["full_date"].min())
# print("Max date:", df["full_date"].max())

# print("Train rows:", len(X_train))
# print("Test rows :", len(X_test))

# print(df[["full_date", "total_revenue"]].head())
# print(df[["full_date", "total_revenue"]].tail())
# print("=" * 50)

# ---------------------------------------
# Train Model
# ---------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------
# Predictions
# ---------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------
# Evaluation
# ---------------------------------------

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")

# ---------------------------------------
# Save Results
# ---------------------------------------

output = test_df = df.loc[test_mask].copy()

output["Actual Revenue"] = y_test.values
output["Predicted Revenue"] = y_pred

# Create output from the complete test dataset
output = test_df.copy()

# Add actual and predicted revenue
output["Actual Revenue"] = y_test.values
output["Predicted Revenue"] = y_pred

print(output.head())

# output.to_csv("forecast_results_1.csv", index=False)

# print("Forecast saved to forecast_results_1.csv")


from tabulate import tabulate

# ---------------------------------------
# Search Forecast (Store ID/Product ID Optional)
# ---------------------------------------

store_input = input("\nEnter Store ID (Press Enter to skip): ").strip()
product_input = input("Enter Product ID (Press Enter to skip): ").strip()

filtered_output = output.copy()

# Filter by Store ID
if store_input:
    filtered_output = filtered_output[
        filtered_output["store_id"] == int(store_input)
    ]

# Filter by Product ID
if product_input:
    filtered_output = filtered_output[
        filtered_output["product_id"] == int(product_input)
    ]

# Check if data exists
if filtered_output.empty:
    print("\nNo records found for the given filters.")
else:

    # Aggregate by Year and Month
    monthly_summary = (
        filtered_output
        .groupby(["year", "month"], as_index=False)
        .agg({
            
            "total_orders": "sum",
            "avg_order_value": "mean",
            "Actual Revenue": "sum",
            "Predicted Revenue": "sum"
        })
    )

    print("\n" + "="*70)
    print("        MONTHLY FORECAST SUMMARY")
    print("="*70)

    if store_input:
        store_name = filtered_output["store_name"].iloc[0]
        print(f"Store   : {store_name} (ID: {store_input})")
    else:
        print("Store   : All Stores")

    if product_input:
        product_name = filtered_output["product_name"].iloc[0]
        print(f"Product : {product_name} (ID: {product_input})")
    else:
        print("Product : All Products")

    print("="*70)
    print()
    print(
        tabulate(
            monthly_summary,
            headers="keys",
            tablefmt="grid",
            showindex=False,
            floatfmt=".2f"
        )
    )

    print(f"\nTotal Months: {len(monthly_summary)}")