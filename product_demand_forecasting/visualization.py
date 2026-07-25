import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Forecast Predictions
# -----------------------------
forecast_df = pd.read_csv("forecast_predictions.csv")

# -----------------------------
# Load Model Metrics
# -----------------------------
metrics_df = pd.read_csv("model_metrics.csv")

# ==========================================================
# 1. Actual vs Predicted Quantity
# ==========================================================
plt.figure(figsize=(10,5))

plt.plot(forecast_df["actual_quantity"], label="Actual Quantity")
plt.plot(forecast_df["predicted_quantity"], label="Predicted Quantity")

plt.title("Actual vs Predicted Product Demand")
plt.xlabel("Sales Records")
plt.ylabel("Quantity")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

# ==========================================================
# 2. Model Comparison (MAE)
# ==========================================================
plt.figure(figsize=(6,5))

plt.bar(metrics_df["Model"], metrics_df["MAE"])

plt.title("Model Comparison (MAE)")
plt.xlabel("Model")
plt.ylabel("MAE")

plt.tight_layout()
plt.savefig("outputs/model_comparison_mae.png")
plt.show()