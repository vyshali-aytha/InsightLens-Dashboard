"""
pipelines/underperforming.py
-----------------------------
Underperforming Product & City Detection System — database-backed version.

This is the notebook `Underperforming_Product_City_Detection_System.ipynb`
converted to a callable pipeline for the API.

ONLY CHANGE FROM THE NOTEBOOK: Section 2 (synthetic data generation) is
replaced by `load_data()`, which pulls dim_date / dim_product / dim_store /
dim_customer / fact_sales straight from the `insightlens` warehouse via
db_utils.py. Every downstream section (3-14: cleaning, aggregation, feature
engineering, trend analysis, z-score detection, Isolation Forest, severity
scoring, recommendations, dashboards, exports) is the same logic as the
notebook, unchanged, because it only depends on column names / schema, not
on how the data arrived (this was explicitly designed into the notebook).

Removed vs. the notebook: the "ground truth" sanity-check cells (comparing
flagged entities against a *planted* list of declining products/cities).
Those only make sense against the synthetic generator's known answer key —
there is no equivalent ground truth in real warehouse data, so those cells
are dropped rather than adapted (nothing to adapt them to).
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # headless — save PNGs instead of plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from db_utils import read_table

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.titleweight"] = "bold"

RNG_SEED = 42
np.random.seed(RNG_SEED)

CONFIG = {
    "rolling_weeks": 4,
    "min_periods_for_trend": 6,
    "trend_pvalue_threshold": 0.10,
    "z_score_threshold": -2.0,
    "iso_forest_contamination": 0.08,
    "iso_forest_random_state": RNG_SEED,
    "severity_weights": {
        "revenue_decline": 0.30,
        "quantity_decline": 0.15,
        "trend_slope": 0.20,
        "z_score": 0.20,
        "isolation_forest": 0.15,
    },
    "risk_bins": [0, 20, 40, 60, 80, 100],
    "risk_labels": ["Healthy", "Low", "Medium", "High", "Critical"],
    "top_n": 20,
}


# =====================================================================
# Section 2 (replaced) — Load Data from the insightlens warehouse
# =====================================================================
def load_data():
    """
    Pulls dim_date, dim_product, dim_store, dim_customer, fact_sales from
    Postgres (schema configured in config/db_config.json) instead of
    generating synthetic data.

    week_index is derived in pandas after the read (it is an analytical
    convenience column the notebook's logic depends on for weekly
    aggregation — it is not a warehouse column, so it cannot come from the
    DB query itself; it is computed the same way the notebook always did,
    just from real dates instead of a generated date range).
    """
    dim_product = read_table(
        "dim_product",
        columns=["product_key", "product_id", "product_name", "category",
                 "subcategory", "company_name", "unit_price"],
    )
    dim_store = read_table(
        "dim_store",
        columns=["store_key", "store_id", "store_name", "city", "state"],
    )
    dim_customer = read_table(
        "dim_customer",
        columns=["customer_key", "customer_id", "customer_name", "city", "segment", "gender"],
    )
    dim_date = read_table(
        "dim_date",
        columns=["date_key", "full_date", "day", "month", "quarter", "year", "weekday"],
    )
    fact_sales = read_table(
        "fact_sales",
        columns=["transaction_id", "customer_key", "product_key", "store_key", "date_key",
                 "quantity", "unit_price", "discount", "gross_amount", "net_amount"],
    )

    dim_date["full_date"] = pd.to_datetime(dim_date["full_date"])
    start = dim_date["full_date"].min()
    dim_date["week_index"] = ((dim_date["full_date"] - start).dt.days // 7).astype(int)

    return dim_date, dim_product, dim_store, dim_customer, fact_sales


# =====================================================================
# Section 4.2 — Reusable aggregation helper
# =====================================================================
def aggregate_performance(df, group_cols, period_col, fill_gaps=True):
    period_cols = [period_col] if period_col == "week_index" else ["year", period_col]
    entity_key = group_cols[0]

    agg = (
        df.groupby(group_cols + period_cols)
        .agg(
            revenue=("net_amount", "sum"),
            quantity=("quantity", "sum"),
            avg_selling_price=("unit_price", "mean"),
            transaction_count=("transaction_id", "count"),
        )
        .reset_index()
    )

    if fill_gaps and period_col == "week_index":
        attr_cols = [c for c in group_cols if c != entity_key]
        entity_attrs = df[group_cols].drop_duplicates(entity_key).set_index(entity_key)

        all_entities = entity_attrs.index.unique()
        all_weeks = np.arange(df["week_index"].min(), df["week_index"].max() + 1)
        full_grid = pd.MultiIndex.from_product([all_entities, all_weeks],
                                                names=[entity_key, "week_index"]).to_frame(index=False)

        agg = full_grid.merge(agg.drop(columns=attr_cols, errors="ignore"),
                               on=[entity_key, "week_index"], how="left")
        for col in ["revenue", "quantity", "transaction_count"]:
            agg[col] = agg[col].fillna(0)
        agg = agg.sort_values([entity_key, "week_index"])
        agg["avg_selling_price"] = agg.groupby(entity_key)["avg_selling_price"].ffill().bfill()

        if attr_cols:
            agg = agg.merge(entity_attrs.reset_index(), on=entity_key, how="left")

    agg = agg.sort_values(group_cols + period_cols)
    agg["revenue_growth_pct"] = agg.groupby(entity_key)["revenue"].pct_change() * 100
    agg["quantity_growth_pct"] = agg.groupby(entity_key)["quantity"].pct_change() * 100
    agg.loc[(agg["revenue"] == 0) & (agg.groupby(entity_key)["revenue"].shift(1) == 0), "revenue_growth_pct"] = 0
    return agg


# =====================================================================
# Section 6.1 — Feature engineering
# =====================================================================
def engineer_features(df, entity_col, peer_col=None, peer_avg_prefix="peer"):
    df = df.sort_values([entity_col, "week_index"]).copy()
    win = CONFIG["rolling_weeks"]

    df["rolling_revenue"] = df.groupby(entity_col)["revenue"].transform(
        lambda s: s.rolling(win, min_periods=2).mean())
    df["rolling_quantity"] = df.groupby(entity_col)["quantity"].transform(
        lambda s: s.rolling(win, min_periods=2).mean())
    df["moving_average_revenue"] = df.groupby(entity_col)["revenue"].transform(
        lambda s: s.rolling(win, min_periods=1).mean())

    df["historical_avg_revenue"] = df.groupby(entity_col)["revenue"].transform(
        lambda s: s.expanding(min_periods=2).mean().shift(1))
    df["historical_std_revenue"] = df.groupby(entity_col)["revenue"].transform(
        lambda s: s.expanding(min_periods=2).std().shift(1))

    df["historical_avg_quantity"] = df.groupby(entity_col)["quantity"].transform(
        lambda s: s.expanding(min_periods=2).mean().shift(1))
    df["revenue_deviation"] = df["revenue"] - df["historical_avg_revenue"]
    df["quantity_deviation"] = df["quantity"] - df["historical_avg_quantity"]

    df["revenue_z_score"] = df["revenue_deviation"] / df["historical_std_revenue"].replace(0, np.nan)
    qty_hist_std = df.groupby(entity_col)["quantity"].transform(
        lambda s: s.expanding(min_periods=2).std().shift(1))
    df["quantity_z_score"] = df["quantity_deviation"] / qty_hist_std.replace(0, np.nan)

    if peer_col is not None:
        peer_avg_rev = df.groupby([peer_col, "week_index"])["revenue"].transform("mean")
        peer_avg_qty = df.groupby([peer_col, "week_index"])["quantity"].transform("mean")
        df[f"{peer_avg_prefix}_avg_revenue"] = peer_avg_rev
        df[f"{peer_avg_prefix}_avg_quantity"] = peer_avg_qty
        df["peer_group_difference"] = df["revenue"] - peer_avg_rev

    return df


def add_performance_score(df, entity_col):
    df = df.copy()
    rev_pct_rank = df.groupby("week_index")["revenue"].rank(pct=True)
    growth_clean = df["revenue_growth_pct"].clip(-100, 100).fillna(0)
    growth_norm = (growth_clean - growth_clean.min()) / (growth_clean.max() - growth_clean.min() + 1e-9)
    df["performance_score"] = (0.6 * rev_pct_rank + 0.4 * growth_norm) * 100
    return df


# =====================================================================
# Section 7 — Rolling trend analysis
# =====================================================================
def fit_trend(group, value_col, min_periods=CONFIG["min_periods_for_trend"]):
    g = group.dropna(subset=[value_col]).sort_values("week_index")
    n = len(g)
    if n < min_periods:
        return pd.Series({
            f"{value_col}_slope": np.nan, f"{value_col}_intercept": np.nan,
            f"{value_col}_r_squared": np.nan, f"{value_col}_p_value": np.nan,
            f"{value_col}_n_periods": n, "sufficient_history": False,
        })
    slope, intercept, r_value, p_value, std_err = stats.linregress(g["week_index"], g[value_col])
    return pd.Series({
        f"{value_col}_slope": slope, f"{value_col}_intercept": intercept,
        f"{value_col}_r_squared": r_value ** 2, f"{value_col}_p_value": p_value,
        f"{value_col}_n_periods": n, "sufficient_history": True,
    })


def compute_trends(weekly_df, entity_col, name_cols=None):
    rev_trend = weekly_df.groupby(entity_col, group_keys=False).apply(
        lambda g: fit_trend(g, "revenue")).reset_index()
    qty_trend = weekly_df.groupby(entity_col, group_keys=False).apply(
        lambda g: fit_trend(g, "quantity")).reset_index()
    qty_trend = qty_trend.drop(columns=["sufficient_history"])

    trend = rev_trend.merge(qty_trend, on=entity_col)
    if name_cols:
        lookup = weekly_df[[entity_col] + name_cols].drop_duplicates(entity_col)
        trend = lookup.merge(trend, on=entity_col)
    return trend


def classify_trend(row, threshold=CONFIG["trend_pvalue_threshold"]):
    if not row["sufficient_history"]:
        return "Insufficient Data"
    if row["revenue_slope"] > 0:
        return "Growing" if row["revenue_p_value"] < threshold else "Stable"
    else:
        return "Significant Decline" if row["revenue_p_value"] < threshold else "Mild/Noisy Decline"


# =====================================================================
# Section 9 — Isolation Forest
# =====================================================================
ISO_FEATURES = ["revenue", "quantity", "revenue_growth_pct", "revenue_z_score", "quantity_z_score"]


def run_isolation_forest(weekly_df, trend_df, entity_col, features=ISO_FEATURES,
                          contamination=CONFIG["iso_forest_contamination"]):
    latest_wk = weekly_df["week_index"].max()
    snap = weekly_df[weekly_df["week_index"] == latest_wk].copy()

    slope_col = "revenue_slope"
    snap = snap.merge(trend_df[[entity_col, slope_col]], on=entity_col, how="left")

    feat_cols = features + [slope_col]
    X = snap[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)  # pct_change() can yield inf when the prior period was 0
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=CONFIG["iso_forest_random_state"],
    )
    raw_pred = model.fit_predict(X_scaled)
    raw_score = model.decision_function(X_scaled)

    snap["iso_anomaly_flag"] = raw_pred == -1
    snap["iso_anomaly_score_raw"] = -raw_score

    return snap, model, feat_cols


# =====================================================================
# Section 10 — Decline Severity Scoring
# =====================================================================
def minmax_norm(s):
    rng = s.max() - s.min()
    if rng == 0 or pd.isna(rng):
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rng


def compute_severity(latest_snapshot, trend_df, iso_snapshot, entity_col):
    df = latest_snapshot.merge(
        trend_df[[entity_col, "revenue_slope", "revenue_p_value", "trend_status"]],
        on=entity_col, how="left"
    )
    df = df.merge(
        iso_snapshot[[entity_col, "iso_anomaly_flag", "iso_anomaly_score_raw"]],
        on=entity_col, how="left"
    )

    w = CONFIG["severity_weights"]

    avg_rev_for_scaling = df["revenue"].where(df["revenue"] > 0, df["revenue"].replace(0, np.nan))
    df["revenue_slope_relative"] = df["revenue_slope"] / avg_rev_for_scaling.abs().clip(lower=1)

    revenue_pct_below_hist = (
        (df["historical_avg_revenue"] - df["revenue"]) / df["historical_avg_revenue"].abs().replace(0, np.nan)
    ).clip(lower=0)
    quantity_pct_below_hist = (
        (df["historical_avg_quantity"] - df["quantity"]) / df["historical_avg_quantity"].abs().replace(0, np.nan)
    ).clip(lower=0)

    revenue_decline_component = minmax_norm(revenue_pct_below_hist.fillna(0))
    quantity_decline_component = minmax_norm(quantity_pct_below_hist.fillna(0))
    slope_component = minmax_norm((-df["revenue_slope_relative"].clip(upper=0)).fillna(0))
    z_component = minmax_norm((-df["revenue_z_score"].clip(upper=0)).fillna(0))
    iso_component = minmax_norm(df["iso_anomaly_score_raw"].fillna(df["iso_anomaly_score_raw"].median()))

    df["decline_severity_score"] = 100 * (
        w["revenue_decline"] * revenue_decline_component +
        w["quantity_decline"] * quantity_decline_component +
        w["trend_slope"] * slope_component +
        w["z_score"] * z_component +
        w["isolation_forest"] * iso_component
    )

    df["risk_level"] = pd.cut(
        df["decline_severity_score"], bins=CONFIG["risk_bins"],
        labels=CONFIG["risk_labels"], include_lowest=True
    )
    return df.sort_values("decline_severity_score", ascending=False)


# =====================================================================
# Section 11 — Explanations & recommendations
# =====================================================================
def explain_product(row):
    reasons = []
    if pd.notna(row.get("revenue_slope")) and row["revenue_slope"] < 0 and row.get("trend_status") == "Significant Decline":
        reasons.append(f"revenue has shown a statistically significant downward trend "
                        f"(slope ~ {row['revenue_slope']:.0f}/week, p={row['revenue_p_value']:.3f})")
    if pd.notna(row.get("revenue_z_score")) and row["revenue_z_score"] < CONFIG["z_score_threshold"]:
        reasons.append(f"current revenue is {abs(row['revenue_z_score']):.1f} standard deviations "
                        f"below its own historical average")
    if row.get("iso_anomaly_flag"):
        reasons.append("multiple KPIs jointly look abnormal versus its peer group (Isolation Forest)")
    if not reasons:
        reasons.append("elevated composite risk score without one single dominant cause")
    return "; ".join(reasons).capitalize() + "."


def recommend_action_product(row):
    if row["risk_level"] in ["Low", "Healthy"]:
        return "Monitor"
    if pd.notna(row.get("revenue_z_score")) and row["revenue_z_score"] < -3:
        return "Investigate Supply Chain"
    if row.get("trend_status") == "Significant Decline" and row["risk_level"] == "Critical":
        return "Replace Product"
    if row.get("trend_status") == "Significant Decline":
        return "Review Pricing"
    if row.get("iso_anomaly_flag"):
        return "Launch Promotion"
    return "Increase Marketing"


def explain_city(row):
    reasons = []
    if pd.notna(row.get("revenue_slope")) and row["revenue_slope"] < 0 and row.get("trend_status") == "Significant Decline":
        reasons.append(f"revenue trend is significantly negative (slope ~ {row['revenue_slope']:.0f}/week)")
    if pd.notna(row.get("revenue_z_score")) and row["revenue_z_score"] < CONFIG["z_score_threshold"]:
        reasons.append(f"current revenue is {abs(row['revenue_z_score']):.1f} std. deviations "
                        f"below the city's own historical average")
    if row.get("iso_anomaly_flag"):
        reasons.append("KPI combination is anomalous versus peer cities in the same state")
    if not reasons:
        reasons.append("elevated composite risk score without one single dominant cause")
    return "; ".join(reasons).capitalize() + "."


def recommend_action_city(row):
    if row["risk_level"] in ["Low", "Healthy"]:
        return "Monitor"
    if row["risk_level"] == "Critical":
        return "Store Audit"
    if row.get("trend_status") == "Significant Decline":
        return "Review Local Demand"
    if row.get("iso_anomaly_flag"):
        return "Investigate Competition"
    return "Increase Advertising"


# =====================================================================
# Plot helpers (Section 5 / 12 — plt.show() -> save PNG for API serving)
# =====================================================================
def _savefig(graphs_dir, name):
    path = os.path.join(graphs_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path


def _build_eda_and_dashboard_graphs(sales_full, product_weekly, city_weekly, product_trends,
                                     product_scored, city_scored, product_weekly_full,
                                     product_recommendations, city_recommendations,
                                     PRODUCT_REPORT_COLS, CITY_REPORT_COLS, graphs_dir):
    graph_files = {}

    # 5.1 Overall daily revenue/quantity trend
    daily = sales_full.groupby("full_date").agg(revenue=("net_amount", "sum"),
                                                 quantity=("quantity", "sum")).reset_index()
    daily["revenue_7d_avg"] = daily["revenue"].rolling(7).mean()
    daily["quantity_7d_avg"] = daily["quantity"].rolling(7).mean()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(daily["full_date"], daily["revenue"], alpha=0.3, color="steelblue", label="Daily revenue")
    axes[0].plot(daily["full_date"], daily["revenue_7d_avg"], color="steelblue", linewidth=2, label="7-day avg")
    axes[0].set_title("Overall Daily Revenue Trend"); axes[0].set_ylabel("Revenue"); axes[0].legend()
    axes[1].plot(daily["full_date"], daily["quantity"], alpha=0.3, color="darkorange", label="Daily quantity")
    axes[1].plot(daily["full_date"], daily["quantity_7d_avg"], color="darkorange", linewidth=2, label="7-day avg")
    axes[1].set_title("Overall Daily Quantity Sold"); axes[1].set_ylabel("Units"); axes[1].legend()
    graph_files["daily_revenue_quantity_trend"] = _savefig(graphs_dir, "daily_revenue_quantity_trend")

    # 5.2 Product/category revenue
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top_products = sales_full.groupby("product_name")["net_amount"].sum().sort_values(ascending=False).head(15)
    sns.barplot(x=top_products.values, y=top_products.index, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Top 15 Products by Total Revenue"); axes[0].set_xlabel("Revenue")
    cat_rev = sales_full.groupby("category")["net_amount"].sum().sort_values(ascending=False)
    sns.barplot(x=cat_rev.values, y=cat_rev.index, ax=axes[1], palette="Greens_r")
    axes[1].set_title("Revenue by Category"); axes[1].set_xlabel("Revenue")
    graph_files["product_category_revenue"] = _savefig(graphs_dir, "product_category_revenue")

    # 5.3 City revenue
    city_rev = sales_full.groupby(["city", "state"])["net_amount"].sum().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=city_rev, x="net_amount", y="city", palette="Purples_r")
    plt.title("Revenue by City"); plt.xlabel("Revenue"); plt.ylabel("City")
    graph_files["city_revenue"] = _savefig(graphs_dir, "city_revenue")

    # 5.4 Monthly / weekly trend
    monthly_trend = sales_full.groupby(["year", "month"]).agg(revenue=("net_amount", "sum")).reset_index()
    monthly_trend["period"] = monthly_trend["year"].astype(str) + "-" + monthly_trend["month"].astype(str).str.zfill(2)
    weekly_trend = sales_full.groupby("week_index").agg(revenue=("net_amount", "sum")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(monthly_trend["period"], monthly_trend["revenue"], marker="o", color="teal")
    axes[0].set_title("Monthly Revenue Trend"); axes[0].tick_params(axis="x", rotation=60)
    axes[1].plot(weekly_trend["week_index"], weekly_trend["revenue"], marker=".", color="crimson")
    axes[1].set_title("Weekly Revenue Trend"); axes[1].set_xlabel("Week Index")
    graph_files["monthly_weekly_revenue_trend"] = _savefig(graphs_dir, "monthly_weekly_revenue_trend")

    # 5.5 Category revenue heatmap
    heat_data = sales_full.pivot_table(index="category", columns="month", values="net_amount", aggfunc="sum")
    plt.figure(figsize=(14, 5))
    sns.heatmap(heat_data, cmap="YlOrRd", annot=False, cbar_kws={"label": "Revenue"})
    plt.title("Category Revenue Heatmap by Month"); plt.xlabel("Month"); plt.ylabel("Category")
    graph_files["category_revenue_heatmap"] = _savefig(graphs_dir, "category_revenue_heatmap")

    # 10.5 Severity distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=product_scored, x="risk_level", order=CONFIG["risk_labels"], ax=axes[0], palette="RdYlGn_r")
    axes[0].set_title("Product Risk Level Distribution")
    sns.countplot(data=city_scored, x="risk_level", order=CONFIG["risk_labels"], ax=axes[1], palette="RdYlGn_r")
    axes[1].set_title("City Risk Level Distribution")
    graph_files["severity_distribution"] = _savefig(graphs_dir, "severity_distribution")

    # 12.1 Product dashboard
    top20_products = product_recommendations.sort_values(
        "decline_severity_score", ascending=False).head(CONFIG["top_n"])
    palette_map = {"Critical": "#8b0000", "High": "#e34a33", "Medium": "#fdae61",
                   "Low": "#a6d96a", "Healthy": "#1a9850"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = top20_products["risk_level"].astype(str).map(palette_map)
    axes[0, 0].barh(top20_products["product_name"][::-1], top20_products["decline_severity_score"][::-1],
                     color=colors[::-1])
    axes[0, 0].set_title("Top 20 Underperforming Products — Severity Ranking", fontsize=11)
    axes[0, 0].set_xlabel("Decline Severity Score")
    cat_perf = product_scored.groupby("category")["decline_severity_score"].mean().sort_values(ascending=False)
    axes[0, 1].bar(cat_perf.index, cat_perf.values, color="indianred")
    axes[0, 1].set_title("Average Severity Score by Category", fontsize=11)
    axes[0, 1].tick_params(axis="x", rotation=30)
    if len(top20_products) > 0:
        worst_pk = top20_products.iloc[0]["product_key"]
        g = product_weekly_full[product_weekly_full["product_key"] == worst_pk].sort_values("week_index")
        axes[1, 0].plot(g["week_index"], g["revenue"], marker="o", markersize=3, color="darkred")
        axes[1, 0].set_title(f"Revenue Trend — {top20_products.iloc[0]['product_name']} (Highest Risk)", fontsize=11)
        axes[1, 0].set_xlabel("Week"); axes[1, 0].set_ylabel("Revenue")
        axes[1, 1].plot(g["week_index"], g["quantity"], marker="o", markersize=3, color="chocolate")
        axes[1, 1].set_title(f"Quantity Trend — {top20_products.iloc[0]['product_name']} (Highest Risk)", fontsize=11)
        axes[1, 1].set_xlabel("Week"); axes[1, 1].set_ylabel("Units Sold")
    graph_files["product_dashboard"] = _savefig(graphs_dir, "product_dashboard")

    # 12.2 City dashboard
    top_cities = city_recommendations.sort_values("decline_severity_score", ascending=False)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors_c = top_cities["risk_level"].astype(str).map(palette_map)
    axes[0, 0].barh(top_cities["city"][::-1], top_cities["decline_severity_score"][::-1], color=colors_c[::-1])
    axes[0, 0].set_title("City Severity Ranking", fontsize=11)
    axes[0, 0].set_xlabel("Decline Severity Score")
    state_perf = city_scored.groupby("state")["decline_severity_score"].mean().sort_values(ascending=False)
    axes[0, 1].bar(state_perf.index, state_perf.values, color="steelblue")
    axes[0, 1].set_title("Average Severity Score by State (Regional Comparison)", fontsize=11)
    axes[0, 1].tick_params(axis="x", rotation=30)
    if len(top_cities) > 0:
        worst_city = top_cities.iloc[0]["city"]
        gc = city_weekly[city_weekly["city"] == worst_city].sort_values("week_index")
        axes[1, 0].plot(gc["week_index"], gc["revenue"], marker="o", markersize=3, color="navy")
        axes[1, 0].set_title(f"Revenue Trend — {worst_city} (Highest Risk)", fontsize=11)
        axes[1, 0].set_xlabel("Week"); axes[1, 0].set_ylabel("Revenue")
        axes[1, 1].plot(gc["week_index"], gc["quantity"], marker="o", markersize=3, color="teal")
        axes[1, 1].set_title(f"Quantity Trend — {worst_city} (Highest Risk)", fontsize=11)
        axes[1, 1].set_xlabel("Week"); axes[1, 1].set_ylabel("Units Sold")
    graph_files["city_dashboard"] = _savefig(graphs_dir, "city_dashboard")

    # 12.3 Supporting distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(product_trends["revenue_slope"].dropna(), bins=30, ax=axes[0], color="mediumpurple")
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set_title("Trend Slope Distribution — Products")
    sns.boxplot(data=product_scored, x="risk_level", y="revenue_z_score",
                order=CONFIG["risk_labels"], ax=axes[1], palette="RdYlGn_r")
    axes[1].set_title("Z-Score Spread by Risk Tier")
    graph_files["trend_zscore_distribution"] = _savefig(graphs_dir, "trend_zscore_distribution")

    return graph_files


# =====================================================================
# Full pipeline entry point
# =====================================================================
def run(csv_dir: str, graphs_dir: str) -> dict:
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # ---- Section 2 (replaced): load from DB ----
    dim_date, dim_product, dim_store, dim_customer, fact_sales = load_data()

    # ---- Section 3: Data Cleaning ----
    fact_sales = fact_sales.drop_duplicates(subset=[c for c in fact_sales.columns if c != "transaction_id"])

    fact_sales["quantity"] = fact_sales.groupby("product_key")["quantity"].transform(lambda s: s.fillna(s.median()))
    fact_sales["quantity"] = fact_sales["quantity"].fillna(fact_sales["quantity"].median())
    recompute_mask = fact_sales["net_amount"].isnull()
    fact_sales.loc[recompute_mask, "net_amount"] = (
        fact_sales.loc[recompute_mask, "gross_amount"] * (1 - fact_sales.loc[recompute_mask, "discount"])
    )

    fact_sales["quantity"] = fact_sales["quantity"].round().astype(int)
    fact_sales["date_key"] = fact_sales["date_key"].astype(int)
    for col in ["unit_price", "discount", "gross_amount", "net_amount"]:
        fact_sales[col] = fact_sales[col].astype(float)
    dim_date["date_key"] = dim_date["date_key"].astype(int)
    dim_product["product_key"] = dim_product["product_key"].astype(int)
    dim_store["store_key"] = dim_store["store_key"].astype(int)

    invalid_price_mask = fact_sales["unit_price"] <= 0
    median_price_by_product = fact_sales[fact_sales["unit_price"] > 0].groupby("product_key")["unit_price"].median()
    fact_sales.loc[invalid_price_mask, "unit_price"] = fact_sales.loc[invalid_price_mask, "product_key"].map(median_price_by_product)
    invalid_qty_mask = fact_sales["quantity"] <= 0
    fact_sales.loc[invalid_qty_mask, "quantity"] = 1
    fact_sales["discount"] = fact_sales["discount"].clip(0, 0.9)
    fact_sales["gross_amount"] = (fact_sales["quantity"] * fact_sales["unit_price"]).round(2)
    fact_sales["net_amount"] = (fact_sales["gross_amount"] * (1 - fact_sales["discount"])).round(2)

    def iqr_outlier_flags(series, k=1.5):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        return (series < lower) | (series > upper)

    fact_sales["is_outlier_net_amount"] = iqr_outlier_flags(fact_sales["net_amount"])

    # ---- Section 4: Aggregation ----
    sales_full = (
        fact_sales
        .merge(dim_product[["product_key", "product_name", "category", "subcategory", "company_name"]],
               on="product_key", how="left")
        .merge(dim_store[["store_key", "store_name", "city", "state"]], on="store_key", how="left")
        .merge(dim_date[["date_key", "full_date", "week_index", "month", "quarter", "year"]],
               on="date_key", how="left")
    )

    product_weekly = aggregate_performance(
        sales_full, group_cols=["product_key", "product_name", "category", "subcategory"],
        period_col="week_index"
    )
    city_weekly = aggregate_performance(sales_full, group_cols=["city", "state"], period_col="week_index")

    stores_per_city = dim_store.groupby("city")["store_key"].nunique().rename("store_count")
    city_weekly = city_weekly.merge(stores_per_city, on="city", how="left")
    city_weekly["avg_revenue_per_store"] = city_weekly["revenue"] / city_weekly["store_count"]
    city_weekly["avg_quantity_per_store"] = city_weekly["quantity"] / city_weekly["store_count"]

    # ---- Section 6: Feature engineering ----
    product_weekly = engineer_features(product_weekly, entity_col="product_key",
                                        peer_col="category", peer_avg_prefix="category")
    city_weekly = engineer_features(city_weekly, entity_col="city", peer_col="state", peer_avg_prefix="state")
    product_weekly = add_performance_score(product_weekly, "product_key")
    city_weekly = add_performance_score(city_weekly, "city")

    # ---- Section 7: Trend analysis ----
    product_trends = compute_trends(
        product_weekly, entity_col="product_key", name_cols=["product_name", "category", "subcategory"]
    )
    product_trends["trend_status"] = product_trends.apply(classify_trend, axis=1)

    city_trends = compute_trends(city_weekly, entity_col="city", name_cols=["state"])
    city_trends["trend_status"] = city_trends.apply(classify_trend, axis=1)

    # ---- Section 8: Z-score outlier detection ----
    Z_THRESH = CONFIG["z_score_threshold"]
    product_weekly["z_flag_revenue"] = product_weekly["revenue_z_score"] < Z_THRESH
    product_weekly["z_flag_quantity"] = product_weekly["quantity_z_score"] < Z_THRESH
    product_weekly["z_flag_any"] = product_weekly["z_flag_revenue"] | product_weekly["z_flag_quantity"]
    city_weekly["z_flag_revenue"] = city_weekly["revenue_z_score"] < Z_THRESH
    city_weekly["z_flag_quantity"] = city_weekly["quantity_z_score"] < Z_THRESH
    city_weekly["z_flag_any"] = city_weekly["z_flag_revenue"] | city_weekly["z_flag_quantity"]

    latest_week = product_weekly["week_index"].max()
    product_latest = product_weekly[product_weekly["week_index"] == latest_week].copy()
    city_latest = city_weekly[city_weekly["week_index"] == latest_week].copy()

    # ---- Section 9: Isolation Forest ----
    product_iso, _, _ = run_isolation_forest(product_weekly, product_trends, entity_col="product_key")
    city_iso, _, _ = run_isolation_forest(city_weekly, city_trends, entity_col="city")

    # ---- Section 10: Severity scoring ----
    product_scored = compute_severity(product_latest, product_trends, product_iso, entity_col="product_key")
    city_scored = compute_severity(city_latest, city_trends, city_iso, entity_col="city")

    # ---- Section 11: Recommendations ----
    product_recommendations = product_scored.copy()
    product_recommendations["explanation"] = product_recommendations.apply(explain_product, axis=1)
    product_recommendations["recommended_action"] = product_recommendations.apply(recommend_action_product, axis=1)
    product_recommendations = product_recommendations.rename(columns={
        "revenue_growth_pct": "revenue_change_pct", "quantity_growth_pct": "quantity_change_pct",
        "revenue_slope": "trend_slope",
    })
    PRODUCT_REPORT_COLS = ["product_name", "category", "revenue", "quantity", "revenue_change_pct",
                            "quantity_change_pct", "trend_slope", "decline_severity_score", "risk_level",
                            "recommended_action", "explanation"]

    city_recommendations = city_scored.copy()
    city_recommendations["explanation"] = city_recommendations.apply(explain_city, axis=1)
    city_recommendations["recommended_action"] = city_recommendations.apply(recommend_action_city, axis=1)
    city_recommendations = city_recommendations.rename(columns={
        "revenue_growth_pct": "growth_pct", "revenue_slope": "trend_slope",
    })
    CITY_REPORT_COLS = ["city", "state", "revenue", "quantity", "growth_pct", "trend_slope",
                         "decline_severity_score", "risk_level", "recommended_action", "explanation"]

    top20_products = product_recommendations.sort_values("decline_severity_score", ascending=False).head(CONFIG["top_n"])
    top_cities = city_recommendations.sort_values("decline_severity_score", ascending=False)

    # ---- Section 13: Method agreement / flag stability ----
    agreement_df = product_scored[["product_key", "product_name"]].copy()
    agreement_df["flagged_by_trend"] = product_scored["product_key"].isin(
        product_trends.loc[product_trends["trend_status"] == "Significant Decline", "product_key"])
    agreement_df["flagged_by_zscore"] = product_scored["product_key"].isin(
        product_latest.loc[product_latest["z_flag_any"], "product_key"])
    agreement_df["flagged_by_isoforest"] = product_scored["product_key"].isin(
        product_iso.loc[product_iso["iso_anomaly_flag"], "product_key"])
    agreement_df["n_methods_agreeing"] = agreement_df[
        ["flagged_by_trend", "flagged_by_zscore", "flagged_by_isoforest"]
    ].sum(axis=1)

    recent_weeks = sorted(product_weekly["week_index"].unique())[-8:]
    recent = product_weekly[product_weekly["week_index"].isin(recent_weeks)].copy()
    flag_stability = (
        recent.groupby("product_key")["z_flag_any"]
        .agg(weeks_flagged="sum", weeks_observed="count")
        .reset_index()
    )
    flag_stability["persistence_pct"] = (flag_stability["weeks_flagged"] / flag_stability["weeks_observed"] * 100).round(1)
    flag_stability = flag_stability.merge(dim_product[["product_key", "product_name"]], on="product_key")

    # ---- Section 12: Dashboards (saved as PNG instead of plt.show()) ----
    graph_files = _build_eda_and_dashboard_graphs(
        sales_full, product_weekly, city_weekly, product_trends, product_scored, city_scored,
        product_weekly, product_recommendations, city_recommendations,
        PRODUCT_REPORT_COLS, CITY_REPORT_COLS, graphs_dir,
    )

    # ---- Section 14: Export deliverables ----
    deliverables = {
        "clean_fact_sales.csv": fact_sales,
        "product_weekly_kpis.csv": product_weekly,
        "city_weekly_kpis.csv": city_weekly,
        "product_trend_analysis.csv": product_trends,
        "city_trend_analysis.csv": city_trends,
        "product_isolation_forest_results.csv": product_iso,
        "city_isolation_forest_results.csv": city_iso,
        "product_severity_scores.csv": product_scored,
        "city_severity_scores.csv": city_scored,
        "ranked_underperforming_products.csv": top20_products,
        "ranked_underperforming_cities.csv": top_cities,
        "product_recommendations.csv": product_recommendations[PRODUCT_REPORT_COLS],
        "city_recommendations.csv": city_recommendations[CITY_REPORT_COLS],
        "method_agreement.csv": agreement_df,
        "flag_stability.csv": flag_stability,
    }
    csv_files = {}
    for filename, df in deliverables.items():
        path = os.path.join(csv_dir, filename)
        df.to_csv(path, index=False)
        csv_files[filename] = path

    return {
        "product_recommendations": product_recommendations[PRODUCT_REPORT_COLS].to_dict(orient="records"),
        "city_recommendations": city_recommendations[CITY_REPORT_COLS].to_dict(orient="records"),
        "top20_products": top20_products[PRODUCT_REPORT_COLS].to_dict(orient="records"),
        "top_cities": top_cities[CITY_REPORT_COLS].to_dict(orient="records"),
        "product_severity_scores": product_scored.drop(columns=["is_outlier_net_amount"], errors="ignore").to_dict(orient="records"),
        "city_severity_scores": city_scored.to_dict(orient="records"),
        "method_agreement": agreement_df.to_dict(orient="records"),
        "flag_stability": flag_stability.to_dict(orient="records"),
        "csv_files": csv_files,
        "graph_files": graph_files,
    }
