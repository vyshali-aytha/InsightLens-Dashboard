import streamlit as st
import pandas as pd
from config import MODELS
from theme import STATUS_COLORS
from utils.data_loader import (
    get_health, get_scores, get_chart_data,
    get_return_dashboard, get_return_breakdown,
    get_discount_products, post_discount_prediction,
)

st.title("Model Dashboard")

active_models = {k: v for k, v in MODELS.items() if v["status"] == "active"}

if not active_models:
    st.warning("No active models yet. Add one in config.py once its API/CSVs are ready.")
    st.stop()

model_key = st.sidebar.selectbox(
    "Model",
    options=list(active_models.keys()),
    format_func=lambda k: active_models[k]["label"],
)
cfg = active_models[model_key]
st.subheader(cfg["label"])


# =============================================================================
# type: risk_scoring — module7-style batch scorer (/health, /scores, /charts/*)
# =============================================================================
def render_risk_scoring(model_key, cfg):
    health = get_health(model_key, cfg["api_base"])
    status = health.get("status", "unknown")
    colors = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Status")
        st.markdown(
            f'<span style="background:{colors["bg"]};color:{colors["fg"]};'
            f'padding:2px 10px;border-radius:999px;font-size:1rem;font-weight:600;">{status}</span>',
            unsafe_allow_html=True,
        )
    c2.metric("Rows scored", health.get("rows_scored", "—"))
    c3.metric("Last refreshed", health.get("last_scored_at") or "—")
    if health.get("last_error"):
        st.caption(health["last_error"])

    st.divider()
    st.markdown("### Scores")
    flagged_only = st.checkbox("Flagged only", value=False)
    limit = st.slider("Rows to show", min_value=10, max_value=500, value=100, step=10)

    scores_df = get_scores(model_key, cfg, flagged_only=flagged_only, limit=limit)
    if scores_df is not None and not scores_df.empty:
        st.dataframe(scores_df, use_container_width=True)
    else:
        st.info("No scores available yet — check the API is running or place a CSV in the model's data folder.")

    st.divider()
    chart_keys = list(cfg["endpoints"].get("charts", {}).keys())
    if chart_keys:
        st.markdown("### Charts")
        chart_titles = {
            "by_method_provider": "Risk by method × provider",
            "by_month": "Seasonality (by month)",
            "calibration": "Calibration",
        }
        tabs = st.tabs([chart_titles.get(k, k) for k in chart_keys])
        for tab, chart_key in zip(tabs, chart_keys):
            with tab:
                chart_df = get_chart_data(model_key, cfg, chart_key)
                if chart_df is None or chart_df.empty:
                    st.info("No chart data available yet.")
                    continue
                _render_risk_chart(chart_key, chart_df)
    else:
        st.caption("No charts configured for this model yet.")


def _render_risk_chart(chart_key, df):
    """Each chart_key has its own shape, so each gets its own rendering
    instead of forcing every chart through the same bar_chart call."""
    import altair as alt

    if chart_key == "by_month":
        # What the user actually wants to see: which months have more
        # transactions AND more failures, not just an abstract rate line.
        # Bars = transaction volume (n_transactions), colored by whether
        # it's a peak-season month; line = actual_fail_rate on its own
        # axis, so both "more traffic" and "more failures" read at a glance.
        df = df.copy()
        df["season"] = df["is_peak_season"].map({True: "Peak (Oct/Nov/Dec)", False: "Regular"})
        base = alt.Chart(df).encode(x=alt.X("month:O", title="Month"))
        bars = base.mark_bar().encode(
            y=alt.Y("n_transactions:Q", title="Transactions"),
            color=alt.Color("season:N", title=None, scale=alt.Scale(
                domain=["Regular", "Peak (Oct/Nov/Dec)"], range=["#a8d1f0", "#0b5fa5"])),
            tooltip=["month", "n_transactions", "n_flagged", "actual_fail_rate"],
        )
        line = base.mark_line(color="crimson", point=True).encode(
            y=alt.Y("actual_fail_rate:Q", title="Actual fail rate", axis=alt.Axis(format="%")),
            tooltip=["month", "actual_fail_rate"],
        )
        chart = alt.layer(bars, line).resolve_scale(y="independent").properties(height=400)
        st.altair_chart(chart, use_container_width=True)
        st.caption("Bars = transaction volume (blue = peak season). Red line = actual failure rate, right axis.")
        st.dataframe(df.drop(columns=["season"]), use_container_width=True)

    elif chart_key == "by_method_provider":
        # Horizontal grouped bars, sorted by mean_risk_score — vertical
        # bars with 12+ combos crushed the method/provider labels into
        # each other; horizontal bars give each label its own full row.
        df = df.copy()
        df["combo"] = df["payment_method"] + " / " + df["payment_provider"]
        order = df.sort_values("mean_risk_score", ascending=False)["combo"].tolist()
        long_df = df.melt(
            id_vars=["combo", "n_transactions"],
            value_vars=["mean_risk_score", "actual_fail_rate"],
            var_name="metric", value_name="value",
        )
        chart = (
            alt.Chart(long_df)
            .mark_bar()
            .encode(
                y=alt.Y("combo:N", title=None, sort=order),
                x=alt.X("value:Q", title="rate"),
                yOffset="metric:N",
                color="metric:N",
                tooltip=["combo", "metric", "value", "n_transactions"],
            )
            .properties(height=alt.Step(18))
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df.drop(columns=["combo"]), use_container_width=True)

    elif chart_key == "calibration":
        # Scatter: predicted (mean_risk_score) vs. observed (actual_fail_rate)
        # per score bin, with a 45° reference line. Points on the line =
        # well-calibrated; points off it = the model over/under-estimates
        # risk in that score range. Point size = n_resolved, so thin/noisy
        # bins (per README's low_sample flag) visibly look less trustworthy.
        df = df.dropna(subset=["actual_fail_rate"]).copy()
        if df.empty:
            st.info("No resolved payments in any bin yet to compare against.")
            return
        max_val = max(df["mean_risk_score"].max(), df["actual_fail_rate"].max()) * 1.05
        diagonal = alt.Chart(pd.DataFrame({"x": [0, max_val], "y": [0, max_val]})).mark_line(
            strokeDash=[4, 4], color="gray"
        ).encode(x="x", y="y")
        points = (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X("mean_risk_score:Q", title="Predicted (mean_risk_score)"),
                y=alt.Y("actual_fail_rate:Q", title="Observed (actual_fail_rate)"),
                size=alt.Size("n_resolved:Q", title="n_resolved"),
                color=alt.Color("low_sample:N", title="Low sample"),
                tooltip=["bin_low", "bin_high", "mean_risk_score", "actual_fail_rate", "n_resolved", "low_sample"],
            )
        )
        st.altair_chart((diagonal + points).properties(height=400), use_container_width=True)
        st.dataframe(df, use_container_width=True)

    else:
        st.dataframe(df, use_container_width=True)



# =============================================================================
# type: return_dashboard — Return Risk's api.py shape (/dashboard + /top-*)
# =============================================================================
def render_return_dashboard(cfg):
    dash = get_return_dashboard(cfg)
    if dash is None:
        st.warning("No data available — check the API is running or place predictions.csv in the model's data folder.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total transactions", dash.get("total_transactions", "—"))
    c2.metric("Predicted returns", dash.get("predicted_returns", "—"))
    c3.metric("Overall return rate", f"{dash.get('overall_return_rate', 0):.1%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Highest-return city", dash.get("highest_return_city", "—"))
    c5.metric("Highest-return product key", dash.get("highest_return_product", "—"))
    c6.metric("Top return reason", dash.get("top_return_reason", "—"))

    st.divider()
    st.markdown("### Breakdowns")
    tabs = st.tabs(["Top returned products", "Top return cities", "Return reasons", "High-risk orders"])
    breakdown_keys = ["top_returned_products", "top_return_cities", "return_reasons", "high_risk_orders"]
    for tab, key in zip(tabs, breakdown_keys):
        with tab:
            df = get_return_breakdown(cfg, key)
            if df is None or df.empty:
                st.info("No data available yet.")
            else:
                st.dataframe(df, use_container_width=True)


# =============================================================================
# type: discount_calculator — Discount Impact's api.py shape
# (POST /predict_discount — a what-if tool, not a precomputed batch result)
# =============================================================================
def render_discount_calculator(cfg):
    products_df = get_discount_products(cfg)
    if products_df is None or products_df.empty:
        st.warning(f"No product list found — place {cfg['products_csv']} in the model's data folder.")
        return

    st.markdown("### Pick a product and discount")
    product_map = dict(zip(products_df["product_id"], products_df["product_name"]))
    product_label = st.selectbox(
        "Product",
        options=products_df["product_id"],
        format_func=lambda pid: f"{pid} — {product_map.get(pid, 'Unknown')}",
    )
    discount = st.select_slider("Current discount (%)", options=cfg["discount_levels"], value=cfg["discount_levels"][0])

    if st.button("Get recommendation", type="primary"):
        result = post_discount_prediction(cfg, product_label, discount)
        if result is None:
            st.error("Discount API isn't reachable, and this tool has no CSV fallback — it's a live what-if calculation.")
        elif "error" in result:
            st.error(f"Request failed: {result['error']}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Current discount", f"{result['current_discount']:.0f}%")
            c2.metric("Recommended discount", f"{result['recommended_discount']:.0f}%")
            c3.metric("Revenue gain", f"{result['revenue_gain']:,.2f}")
            st.info(f"**{result['recommended_action']}** — expected revenue at recommended discount: {result['recommended_expected_revenue']:,.2f} (vs {result['current_expected_revenue']:,.2f} now)")
            with st.expander("Full response"):
                st.json(result)


# --- Dispatch on type ---
model_type = cfg.get("type")
if model_type == "risk_scoring":
    render_risk_scoring(model_key, cfg)
elif model_type == "return_dashboard":
    render_return_dashboard(cfg)
elif model_type == "discount_calculator":
    render_discount_calculator(cfg)
else:
    st.info("This model doesn't have a render type configured yet.")
