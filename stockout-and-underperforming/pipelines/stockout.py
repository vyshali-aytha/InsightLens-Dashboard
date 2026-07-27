"""
pipelines/stockout.py
----------------------
Stockout & Reorder Prediction System — database-backed version.

Converted from `Stockout_Reorder_Prediction_System.ipynb`.

ONLY CHANGE FROM THE NOTEBOOK: Section 2 (data loading) now reads
dim_product, dim_store, and fact_inventory straight from the `insightlens`
warehouse via db_utils.py, instead of from CSV / the synthetic generator.

`fact_forecast` (forecasted_daily_demand) has NO corresponding table in
updated_database.sql / the insightlens schema — no such table is created
here either, per the task constraints. Instead, `forecasted_daily_demand`
is now calculated dynamically on every pipeline run from real sales
history in `fact_sales` (SUM(quantity) / number of distinct selling days,
i.e. a historical/rolling average daily demand per product+store), via
`_calculate_forecast()` below. The resulting dataframe is saved to
`outputs/csv/fact_forecast.csv` on every run and feeds the rest of the
pipeline exactly as the old random mock did — no downstream logic needed
to change since the dataframe schema (date_key, product_key, store_key,
forecasted_daily_demand) is unchanged. Swap `_calculate_forecast()` for a
real Module 2 forecasting query the moment one exists — nothing else
needs to change.

Everything else (Sections 3-14: cleaning, EDA, feature engineering,
rule-based baseline, ML training/evaluation, recommendation engine,
dashboards, exports) is the same logic as the notebook, unchanged.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    mean_absolute_error
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from db_utils import read_table

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CONFIG = {
    "N_DAYS_STOCKOUT": 7,
    "LEAD_TIME_DAYS": 5,
    "SAFETY_STOCK_DAYS": 3,
    "RISK_THRESHOLDS": {"Critical": 2, "High": 5, "Medium": 10},
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": RANDOM_STATE,
}


# =====================================================================
# Section 2 (replaced) — Load Data from the insightlens warehouse
# =====================================================================
def _calculate_forecast(fact_inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates forecasted_daily_demand from real sales history in
    `fact_sales`, replacing the previous random mock. No fact_forecast
    table exists in the warehouse schema, so nothing is written back to
    the database here — this dataframe is (re)built in-memory on every
    pipeline run.

    Forecast formula (deterministic, no ML, no forecasting libraries):

        forecasted_daily_demand = SUM(quantity) / number_of_selling_days

    computed per (product_key, store_key) from `fact_sales`, i.e. the
    historical/rolling average daily quantity sold for that product at
    that store. This value is then broadcast across every
    (date_key, product_key, store_key) combination present in
    `fact_inventory`, since that is the grain the rest of the pipeline
    (and its downstream 7d/30d rolling-demand features) already expects.

    Product/store combinations with no sales history yet (e.g. brand-new
    listings) fall back to the overall average daily demand across all
    products/stores that do have sales history, rather than a random
    number — if there is no sales history anywhere, the fallback is 0.
    """
    fact_sales = read_table(
        "fact_sales",
        columns=["product_key", "store_key", "date_key", "quantity"],
    )

    sales_agg = (
        fact_sales.groupby(["product_key", "store_key"])
        .agg(total_quantity=("quantity", "sum"), selling_days=("date_key", "nunique"))
        .reset_index()
    )
    sales_agg["forecasted_daily_demand"] = (
        sales_agg["total_quantity"] / sales_agg["selling_days"].replace(0, np.nan)
    )

    overall_avg_demand = sales_agg["forecasted_daily_demand"].mean()
    if pd.isna(overall_avg_demand):
        overall_avg_demand = 0.0

    fc = fact_inventory[["date_key", "product_key", "store_key"]].drop_duplicates().copy()
    fc = fc.merge(
        sales_agg[["product_key", "store_key", "forecasted_daily_demand"]],
        on=["product_key", "store_key"], how="left",
    )
    fc["forecasted_daily_demand"] = fc["forecasted_daily_demand"].fillna(overall_avg_demand).round(2)
    return fc


def load_data():
    dim_product = read_table(
        "dim_product",
        columns=["product_key", "product_id", "product_name", "category",
                 "subcategory", "company_name", "unit_price"],
    )
    dim_store = read_table(
        "dim_store",
        columns=["store_key", "store_id", "store_name", "city", "state"],
    )
    dim_date = read_table(
        "dim_date",
        columns=["date_key", "full_date", "day", "month", "quarter", "year", "weekday"],
    )
    fact_inventory = read_table(
        "fact_inventory",
        columns=["inventory_id", "product_key", "store_key", "date_key",
                  "stock_capacity", "reorder_level", "available_stock", "inventory_status"],
    )

    fact_forecast = _calculate_forecast(fact_inventory)

    return dim_product, dim_store, dim_date, fact_inventory, fact_forecast


def _savefig(graphs_dir, name):
    path = os.path.join(graphs_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path


def classify_risk(days):
    t = CONFIG["RISK_THRESHOLDS"]
    if days <= t["Critical"]:
        return "Critical Risk"
    elif days <= t["High"]:
        return "High Risk"
    elif days <= t["Medium"]:
        return "Medium Risk"
    else:
        return "Low Risk"


def check_ml_readiness(inventory_df, min_snapshots_per_combo=10, min_combos=20):
    snapshot_counts = inventory_df.groupby(["product_key", "store_key"])["date_key"].nunique()
    combos_with_enough_history = (snapshot_counts >= min_snapshots_per_combo).sum()
    ready = combos_with_enough_history >= min_combos
    return ready, combos_with_enough_history, snapshot_counts


def make_labels(group, n):
    group = group.sort_values("full_date").reset_index(drop=True)
    dates = group["full_date"].values
    stock = group["available_stock"].values
    status = group["inventory_status"].values
    labels = np.zeros(len(group), dtype=float)
    for i in range(len(group)):
        window_end = dates[i] + np.timedelta64(n, "D")
        future_mask = (dates > dates[i]) & (dates <= window_end)
        if future_mask.any():
            will_stockout = (stock[future_mask] <= 0) | (status[future_mask] == "Out of Stock")
            labels[i] = int(will_stockout.any())
        else:
            labels[i] = np.nan
    group["stockout_within_n_days"] = labels
    return group


def actual_days_to_stockout(group):
    group = group.sort_values("full_date").reset_index(drop=True)
    dates = group["full_date"].values
    stock = group["available_stock"].values
    actual = np.full(len(group), np.nan)
    for i in range(len(group)):
        future_zero = np.where((stock[i:] <= 0))[0]
        if len(future_zero) > 0:
            actual[i] = (dates[i + future_zero[0]] - dates[i]) / np.timedelta64(1, "D")
    group["actual_days_to_stockout"] = actual
    return group


def priority_level(row, ml_ready):
    if ml_ready and not np.isnan(row["stockout_probability"]):
        if row["stockout_probability"] >= 0.75 or row["risk_category"] == "Critical Risk":
            return "Critical"
        elif row["stockout_probability"] >= 0.5 or row["risk_category"] == "High Risk":
            return "High"
        elif row["stockout_probability"] >= 0.25 or row["risk_category"] == "Medium Risk":
            return "Medium"
        return "Low"
    else:
        return row["risk_category"].replace(" Risk", "")


def run(csv_dir: str, graphs_dir: str) -> dict:
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # ---- Section 2 (replaced): load from DB ----
    dim_product, dim_store, dim_date, fact_inventory, fact_forecast = load_data()

    # Persist the freshly-calculated forecast every run, overwriting the
    # previous CSV, so it always reflects the current database contents.
    fact_forecast.to_csv(os.path.join(csv_dir, "fact_forecast.csv"), index=False)

    # ---- Section 3: Data Cleaning ----
    fact_inventory = fact_inventory.drop_duplicates()
    fact_forecast = fact_forecast.drop_duplicates()

    key_cols = ["date_key", "product_key", "store_key"]
    fact_inventory = fact_inventory.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last")

    for col in ["date_key", "product_key", "store_key", "stock_capacity", "reorder_level", "available_stock"]:
        fact_inventory[col] = pd.to_numeric(fact_inventory[col], errors="coerce")
    for col in ["date_key", "product_key", "store_key", "forecasted_daily_demand"]:
        fact_forecast[col] = pd.to_numeric(fact_forecast[col], errors="coerce")

    for col in ["available_stock", "reorder_level", "stock_capacity"]:
        fact_inventory[col] = (
            fact_inventory.groupby(["product_key", "store_key"])[col].transform(lambda s: s.fillna(s.median()))
        )
        fact_inventory[col] = fact_inventory[col].fillna(fact_inventory[col].median())
    fact_inventory["inventory_status"] = fact_inventory["inventory_status"].fillna("Unknown")

    fact_forecast["forecasted_daily_demand"] = (
        fact_forecast.groupby(["product_key", "store_key"])["forecasted_daily_demand"]
        .transform(lambda s: s.fillna(s.median()))
    )
    fact_forecast["forecasted_daily_demand"] = fact_forecast["forecasted_daily_demand"].fillna(
        fact_forecast["forecasted_daily_demand"].median()
    )

    fact_inventory = fact_inventory.dropna(subset=key_cols)

    fact_inventory["available_stock"] = fact_inventory["available_stock"].clip(lower=0)
    fact_inventory["available_stock"] = fact_inventory[["available_stock", "stock_capacity"]].min(axis=1)
    fact_forecast["forecasted_daily_demand"] = fact_forecast["forecasted_daily_demand"].clip(lower=0)

    q1, q3 = fact_forecast["forecasted_daily_demand"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    fact_forecast["is_demand_outlier"] = fact_forecast["forecasted_daily_demand"] > upper_bound

    # ---- Section 4: EDA snapshot ----
    inv_enriched = fact_inventory.merge(dim_product, on="product_key", how="left").merge(dim_store, on="store_key", how="left")
    latest_snapshot = inv_enriched.sort_values("date_key").groupby(["product_key", "store_key"]).tail(1)

    # ---- Section 5: Feature engineering (master join) ----
    df = (
        fact_inventory
        .merge(fact_forecast, on=["date_key", "product_key", "store_key"], how="inner")
        .merge(dim_product, on="product_key", how="left")
        .merge(dim_store, on="store_key", how="left")
    )
    df = df.rename(columns={"company_name": "supplier"})
    df["full_date"] = pd.to_datetime(df["date_key"], format="%Y%m%d")
    df = df.sort_values(["product_key", "store_key", "full_date"]).reset_index(drop=True)

    grp = df.groupby(["product_key", "store_key"])
    df["days_to_stockout"] = np.where(df["forecasted_daily_demand"] > 0,
                                       df["available_stock"] / df["forecasted_daily_demand"], np.inf)
    df["stock_utilization_pct"] = np.where(df["stock_capacity"] > 0,
                                            (df["available_stock"] / df["stock_capacity"]) * 100, 0)
    df["remaining_capacity"] = df["stock_capacity"] - df["available_stock"]
    df["demand_to_stock_ratio"] = np.where(df["available_stock"] > 0,
                                            df["forecasted_daily_demand"] / df["available_stock"],
                                            df["forecasted_daily_demand"])
    df["rolling_7d_demand"] = grp["forecasted_daily_demand"].transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    df["rolling_30d_demand"] = grp["forecasted_daily_demand"].transform(lambda s: s.rolling(window=30, min_periods=1).mean())
    df["avg_daily_demand"] = grp["forecasted_daily_demand"].transform(lambda s: s.expanding(min_periods=1).mean())
    df["demand_volatility"] = grp["forecasted_daily_demand"].transform(lambda s: s.rolling(window=7, min_periods=1).std()).fillna(0)
    df["inventory_buffer"] = np.where(df["forecasted_daily_demand"] > 0,
                                       (df["available_stock"] - df["reorder_level"]) / df["forecasted_daily_demand"], 0)
    safety_stock_units = df["avg_daily_demand"] * CONFIG["SAFETY_STOCK_DAYS"]
    df["safety_stock_indicator"] = (df["available_stock"] < (df["reorder_level"] + safety_stock_units)).astype(int)

    df["stock_prev"] = grp["available_stock"].shift(1)
    df["is_restock_event"] = (df["available_stock"] > df["stock_prev"]).fillna(False)

    def _days_since_restock(sub):
        days = np.zeros(len(sub))
        counter = 0
        for i, restocked in enumerate(sub["is_restock_event"].values):
            counter = 0 if restocked else counter + 1
            days[i] = counter
        return pd.Series(days, index=sub.index)

    df["days_since_last_restock"] = grp.apply(_days_since_restock).reset_index(level=[0, 1], drop=True)
    df["inventory_turnover_estimate"] = np.where(df["available_stock"] > 0,
                                                  df["forecasted_daily_demand"] / df["available_stock"], 0)
    status_order = {"Out of Stock": 0, "Critical": 1, "Low Stock": 2, "Available": 3, "Unknown": 2}
    df["inventory_status_encoded"] = df["inventory_status"].map(status_order).fillna(2).astype(int)

    util_score = df["stock_utilization_pct"].clip(0, 100)
    buffer_score = df["inventory_buffer"].clip(lower=0).clip(upper=30) / 30 * 100
    volatility_penalty = (df["demand_volatility"] / (df["avg_daily_demand"] + 1e-6)).clip(0, 1) * 100
    df["inventory_health_score"] = (0.4 * util_score + 0.4 * buffer_score - 0.2 * volatility_penalty).clip(0, 100)
    df = df.drop(columns=["stock_prev", "is_restock_event"])

    # ---- Section 6: Rule-based baseline ----
    df["risk_category"] = df["days_to_stockout"].apply(classify_risk)
    df["recommended_reorder_date"] = df["full_date"] + pd.to_timedelta(
        (df["days_to_stockout"] - CONFIG["LEAD_TIME_DAYS"]).clip(lower=0), unit="D"
    )
    safety_stock_units = df["avg_daily_demand"] * CONFIG["SAFETY_STOCK_DAYS"]
    df["reorder_quantity"] = (
        df["forecasted_daily_demand"] * CONFIG["LEAD_TIME_DAYS"] + safety_stock_units - df["available_stock"]
    ).clip(lower=0).round().astype(int)

    # ---- Section 7: ML layer ----
    ML_READY, n_ready_combos, snapshot_counts = check_ml_readiness(fact_inventory)

    results_df = pd.DataFrame()
    importances = pd.Series(dtype=float)
    BEST_MODEL_NAME = None
    BEST_MODEL = None
    feature_cols = []
    scaler = None
    mae = None

    if ML_READY:
        N = CONFIG["N_DAYS_STOCKOUT"]
        df = pd.concat([make_labels(g, N) for _, g in df.groupby(["product_key", "store_key"])], ignore_index=True)
        df_ml = df.dropna(subset=["stockout_within_n_days"]).copy()
        df_ml["stockout_within_n_days"] = df_ml["stockout_within_n_days"].astype(int)

        categorical_features = ["category", "subcategory", "supplier", "city", "state"]
        numeric_features = [
            "available_stock", "forecasted_daily_demand", "reorder_level", "stock_capacity",
            "inventory_status_encoded", "demand_volatility", "rolling_7d_demand", "rolling_30d_demand",
            "avg_daily_demand", "inventory_buffer", "remaining_capacity", "stock_utilization_pct",
            "demand_to_stock_ratio", "safety_stock_indicator", "days_since_last_restock",
            "inventory_turnover_estimate", "inventory_health_score",
        ]
        for col in categorical_features:
            le = LabelEncoder()
            df_ml[col + "_enc"] = le.fit_transform(df_ml[col].astype(str))
            class_map = {cls: idx for idx, cls in enumerate(le.classes_)}
            df[col + "_enc"] = df[col].astype(str).map(class_map).fillna(-1).astype(int)

        feature_cols = numeric_features + [c + "_enc" for c in categorical_features]
        X = df_ml[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df_ml["stockout_within_n_days"]

        if y.nunique() < 2:
            ML_READY = False
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=CONFIG["TEST_SIZE"], random_state=CONFIG["RANDOM_STATE"], stratify=y
            )
            imbalance_ratio = y_train.value_counts(normalize=True)
            use_smote = HAS_SMOTE and (imbalance_ratio.min() < 0.35)
            if use_smote:
                smote = SMOTE(random_state=CONFIG["RANDOM_STATE"])
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                class_weight = None
            else:
                X_train_res, y_train_res = X_train, y_train
                class_weight = "balanced"

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_res)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, class_weight=class_weight, random_state=CONFIG["RANDOM_STATE"]
                ).fit(X_train_scaled, y_train_res),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200, class_weight=class_weight, random_state=CONFIG["RANDOM_STATE"], n_jobs=-1
                ).fit(X_train_res, y_train_res),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=CONFIG["RANDOM_STATE"]
                ).fit(X_train_res, y_train_res),
            }

            # Hyperparameter tuning (Random Forest)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
            }
            rf_search = GridSearchCV(
                estimator=RandomForestClassifier(class_weight=class_weight, random_state=CONFIG["RANDOM_STATE"], n_jobs=-1),
                param_grid=param_grid, scoring="recall",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG["RANDOM_STATE"]),
                n_jobs=-1, verbose=0,
            )
            rf_search.fit(X_train_res, y_train_res)
            models["Random Forest (Tuned)"] = rf_search.best_estimator_

            results = []
            for name, model in models.items():
                X_eval = X_test_scaled if name == "Logistic Regression" else X_test
                y_pred = model.predict(X_eval)
                y_proba = model.predict_proba(X_eval)[:, 1]
                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                    "ROC AUC": roc_auc_score(y_test, y_proba),
                })
            results_df = pd.DataFrame(results).sort_values("Recall", ascending=False).reset_index(drop=True)
            BEST_MODEL_NAME = results_df.iloc[0]["Model"]
            BEST_MODEL = models[BEST_MODEL_NAME]

            X_eval = X_test_scaled if BEST_MODEL_NAME == "Logistic Regression" else X_test
            y_pred = BEST_MODEL.predict(X_eval)
            y_proba = BEST_MODEL.predict_proba(X_eval)[:, 1]

            # Confusion matrix / ROC / PR curve graph
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                        xticklabels=["No Stockout", "Stockout"], yticklabels=["No Stockout", "Stockout"])
            axes[0].set_title(f"Confusion Matrix — {BEST_MODEL_NAME}")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}", color="darkorange")
            axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            axes[1].set_title("ROC Curve"); axes[1].legend()
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            axes[2].plot(rec, prec, color="teal")
            axes[2].set_title("Precision-Recall Curve")
            model_eval_graph = _savefig(graphs_dir, "model_evaluation")

            # Regression evaluation (MAE)
            df_reg = pd.concat([actual_days_to_stockout(g) for _, g in df.groupby(["product_key", "store_key"])], ignore_index=True)
            reg_eval = df_reg.dropna(subset=["actual_days_to_stockout"]).copy()
            if len(reg_eval) > 0:
                reg_eval["predicted_days_to_stockout"] = reg_eval["days_to_stockout"].replace(
                    np.inf, reg_eval["actual_days_to_stockout"].max())
                mae = mean_absolute_error(reg_eval["actual_days_to_stockout"], reg_eval["predicted_days_to_stockout"])

            # Feature importance
            if hasattr(BEST_MODEL, "feature_importances_"):
                importances = pd.Series(BEST_MODEL.feature_importances_, index=feature_cols).sort_values(ascending=False)
            else:
                importances = pd.Series(np.abs(BEST_MODEL.coef_[0]), index=feature_cols).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(9, 8))
            importances.head(15).sort_values().plot(kind="barh", ax=ax, color="mediumpurple")
            ax.set_title(f"Feature Importance — {BEST_MODEL_NAME}")
            feature_importance_graph = _savefig(graphs_dir, "feature_importance")

    # ---- Business Recommendation Engine ----
    latest = df.sort_values("full_date").groupby(["product_key", "store_key"]).tail(1).copy()
    if ML_READY:
        X_latest = latest[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_latest_eval = scaler.transform(X_latest) if BEST_MODEL_NAME == "Logistic Regression" else X_latest
        latest["stockout_probability"] = BEST_MODEL.predict_proba(X_latest_eval)[:, 1]
    else:
        latest["stockout_probability"] = np.nan

    latest["priority_level"] = latest.apply(lambda row: priority_level(row, ML_READY), axis=1)

    recommendation_table = latest[[
        "product_name", "store_name", "available_stock", "forecasted_daily_demand",
        "days_to_stockout", "stockout_probability", "risk_category",
        "recommended_reorder_date", "reorder_quantity", "priority_level"
    ]].rename(columns={
        "product_name": "Product", "store_name": "Store", "available_stock": "Available Stock",
        "forecasted_daily_demand": "Forecast Demand", "days_to_stockout": "Days Until Stockout",
        "stockout_probability": "Stockout Probability", "risk_category": "Risk Category",
        "recommended_reorder_date": "Recommended Reorder Date", "reorder_quantity": "Recommended Reorder Quantity",
        "priority_level": "Priority Level",
    }).sort_values("Priority Level").reset_index(drop=True)

    top20_risk = recommendation_table.sort_values(
        "Stockout Probability" if ML_READY else "Days Until Stockout", ascending=not ML_READY
    ).head(20)

    store_risk = latest.groupby("store_name").agg(
        avg_days_to_stockout=("days_to_stockout", "mean"),
        critical_count=("risk_category", lambda s: (s == "Critical Risk").sum()),
        combos=("product_key", "count"),
    ).sort_values("critical_count", ascending=False)

    category_risk = latest.groupby("category").agg(
        avg_days_to_stockout=("days_to_stockout", "mean"),
        critical_pct=("risk_category", lambda s: round((s == "Critical Risk").mean() * 100, 1)),
        combos=("product_key", "count"),
    ).sort_values("critical_pct", ascending=False)

    health_summary = latest.groupby("risk_category").agg(
        combos=("product_key", "count"),
        avg_health_score=("inventory_health_score", "mean"),
        avg_reorder_qty=("reorder_quantity", "mean"),
    ).sort_values("avg_health_score")

    # ---- Dashboard graphs ----
    graph_files = {}
    order = latest_snapshot["inventory_status"].value_counts().index
    fig, ax = plt.subplots()
    sns.countplot(data=latest_snapshot, x="inventory_status", order=order, ax=ax, palette="viridis")
    ax.set_title("Inventory Status Distribution (latest snapshot)")
    graph_files["inventory_status_distribution"] = _savefig(graphs_dir, "inventory_status_distribution")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.countplot(data=latest, x="priority_level", order=["Critical", "High", "Medium", "Low"],
                  palette="rocket", ax=axes[0, 0])
    axes[0, 0].set_title("Stockout Risk / Priority Distribution")
    latest.groupby("store_name")["risk_category"].apply(
        lambda s: (s == "Critical Risk").mean() * 100).sort_values().plot(kind="barh", ax=axes[0, 1], color="firebrick")
    axes[0, 1].set_title("% Critical Risk by Store")
    latest.groupby("category")["risk_category"].apply(
        lambda s: (s == "Critical Risk").mean() * 100).sort_values().plot(kind="barh", ax=axes[1, 0], color="darkslateblue")
    axes[1, 0].set_title("% Critical Risk by Category")
    finite_days = latest.loc[np.isfinite(latest["days_to_stockout"]), "days_to_stockout"]
    sns.histplot(finite_days.clip(upper=60), bins=30, kde=True, ax=axes[1, 1], color="teal")
    axes[1, 1].set_title("Days Until Stockout — Distribution (clipped at 60 days)")
    graph_files["risk_dashboard"] = _savefig(graphs_dir, "risk_dashboard")

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        latest["forecasted_daily_demand"], latest["available_stock"],
        c=(latest["stockout_probability"] if ML_READY else latest["inventory_status_encoded"]),
        cmap="RdYlGn_r" if ML_READY else "RdYlGn", alpha=0.7, s=40
    )
    ax.set_xlabel("Forecasted Daily Demand"); ax.set_ylabel("Available Stock")
    ax.set_title("Demand vs. Available Stock")
    plt.colorbar(scatter, ax=ax)
    graph_files["demand_vs_stock"] = _savefig(graphs_dir, "demand_vs_stock")

    if ML_READY:
        graph_files["model_evaluation"] = model_eval_graph
        graph_files["feature_importance"] = feature_importance_graph

    # ---- Export deliverables ----
    csv_files = {}

    def _save(name, dframe, index=False):
        path = os.path.join(csv_dir, name)
        dframe.to_csv(path, index=index)
        csv_files[name] = path

    _save("clean_dataset_inventory.csv", fact_inventory)
    _save("engineered_dataset.csv", df)
    _save("rule_based_predictions.csv", df[["product_key", "store_key", "full_date", "days_to_stockout",
                                             "risk_category", "recommended_reorder_date", "reorder_quantity"]])
    if ML_READY:
        _save("ml_stockout_probability.csv", latest[["product_key", "store_key", "stockout_probability"]])
        _save("model_evaluation_metrics.csv", results_df)
        _save("feature_importance.csv", importances.reset_index().rename(columns={"index": "feature", 0: "importance"}))
    _save("final_recommendation_table.csv", recommendation_table)
    _save("top20_highest_risk.csv", top20_risk)
    _save("top_stores_by_risk.csv", store_risk, index=True)
    _save("category_risk_summary.csv", category_risk, index=True)
    _save("inventory_health_summary.csv", health_summary, index=True)

    return {
        "ml_ready": bool(ML_READY),
        "best_model": BEST_MODEL_NAME,
        "model_evaluation_metrics": results_df.to_dict(orient="records") if len(results_df) else [],
        "rule_based_mae_days": mae,
        "recommendation_table": recommendation_table.to_dict(orient="records"),
        "top20_highest_risk": top20_risk.to_dict(orient="records"),
        "store_risk": store_risk.reset_index().to_dict(orient="records"),
        "category_risk": category_risk.reset_index().to_dict(orient="records"),
        "health_summary": health_summary.reset_index().to_dict(orient="records"),
        "csv_files": csv_files,
        "graph_files": graph_files,
        "notes": {
            "fact_forecast": (
                "No fact_forecast table exists in the insightlens schema, so none was created. "
                "forecasted_daily_demand is instead calculated on every run from real fact_sales "
                "history (SUM(quantity) / number of selling days per product+store) and saved to "
                "outputs/csv/fact_forecast.csv. Replace pipelines/stockout.py:_calculate_forecast() "
                "with a real Module 2 forecasting query once one is available."
            )
        },
    }
