import streamlit as st
import pandas as pd
import plotly.express as px
from config import MODELS
from theme import STATUS_COLORS
from utils.data_loader import (
    get_health, get_scores, get_chart_data,
    get_return_dashboard, get_return_breakdown, get_return_predictions,
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
RETURN_CHART_COLORS = ["#8DA9C4", "#B8A1C8", "#9BC5B0", "#E7B98E", "#A7C7C7", "#D6A5A5"]

RETURN_DASHBOARD_CSS = """
<style>
.return-kpi {
    min-height: 132px; padding: 18px 20px; border: 1px solid var(--accent);
    border-left: 5px solid var(--accent); border-radius: 14px;
    background: var(--tint); box-shadow: 0 5px 16px rgba(20, 31, 48, .05);
    transition: transform .16s ease, box-shadow .16s ease;
}
.return-kpi:hover { transform: translateY(-2px); box-shadow: 0 9px 22px rgba(20, 31, 48, .09); }
.return-kpi__label { color: #52606D; font-size: .78rem; font-weight: 700; letter-spacing: .045em; text-transform: uppercase; }
.return-kpi__value { color: #1B2430; font-size: 1.7rem; font-weight: 750; line-height: 1.35; overflow-wrap: anywhere; }
.return-kpi__sub { color: #6B7280; font-size: .82rem; }
.return-insight { padding: 14px 16px; background: #FFFFFF; border: 1px solid #E2E5EA; border-radius: 10px; margin-bottom: 10px; color: #394150; }
</style>
"""


def _return_product_lookup():
    """Use the project product dimension instead of presenting raw keys."""
    products = get_discount_products(MODELS["discount_impact"])
    if products is None or products.empty:
        return {}
    lookup = products.copy()
    lookup["product_key"] = lookup["product_id"].astype(str).str.extract(r"(\d+)$")[0].astype(int).astype(str)
    return lookup.dropna(subset=["product_key"]).set_index("product_key")["product_name"].to_dict()


def _return_product_name(value, lookup):
    key = str(value).replace(".0", "")
    return lookup.get(key, f"Product {key}")


def _return_kpi(label, value, subtitle, accent, tint):
    st.markdown(
        f'<div class="return-kpi" style="--accent:{accent};--tint:{tint};">'
        f'<div class="return-kpi__label">{label}</div>'
        f'<div class="return-kpi__value">{value}</div>'
        f'<div class="return-kpi__sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def _return_flag(series):
    return series.astype(str).str.strip().str.lower().isin(["yes", "true", "1"])


def _return_filters(df):
    filtered = df.copy()
    filter_keys = ["return_customer_city", "return_category", "return_company_name", "return_status"]
    with st.expander("Filters", expanded=False):
        st.caption("Refine the selected model output. All dashboard elements update automatically.")
        first_row = st.columns(2)
        second_row = st.columns(2)
        controls = [
            (first_row[0], "customer_city", "City"),
            (first_row[1], "category", "Category"),
            (second_row[0], "company_name", "Company"),
        ]
        for container, column, label in controls:
            if column in filtered.columns:
                choices = sorted(filtered[column].dropna().astype(str).unique())
                with container:
                    selected = st.multiselect(label, choices, key=f"return_{column}")
                if selected:
                    filtered = filtered[filtered[column].astype(str).isin(selected)]
        if "predicted_return" in filtered.columns:
            statuses = sorted(filtered["predicted_return"].dropna().astype(str).unique())
            with second_row[1]:
                selected_status = st.multiselect("Return status", statuses, default=statuses, key="return_status")
            filtered = filtered[filtered["predicted_return"].astype(str).isin(selected_status)]

        reset_column = st.columns([3, 1])[1]
        with reset_column:
            reset_filters = st.button("Reset Filters", key="reset_return_filters", use_container_width=True)
        if reset_filters:
            for key in filter_keys:
                st.session_state.pop(key, None)
            st.rerun()
    return filtered


def _return_chart_layout(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="white",
        plot_bgcolor="white", font=dict(color="#1B2430"), legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EDF0F3", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def _highlight_return_risk(row):
    colors = {"high": "background-color: #FBECEE; color: #8A3037;", "medium": "background-color: #FFF6E8; color: #7A5500;"}
    style = colors.get(str(row.get("Risk Level", "")).lower(), "")
    return [style] * len(row)


def render_return_dashboard(cfg):
    # Keep the existing /dashboard request available for the service-backed
    # summary, while the model's exported prediction output powers filtering.
    dash = get_return_dashboard(cfg)
    predictions = get_return_predictions(cfg)
    if dash is None and (predictions is None or predictions.empty):
        st.warning("No data available — check the API is running or place predictions.csv in the model's data folder.")
        return
    if predictions is None or predictions.empty:
        st.info("The service summary is available, but transaction-level predictions are needed for interactive analysis.")
        return

    st.markdown(RETURN_DASHBOARD_CSS, unsafe_allow_html=True)
    raw = predictions.copy()
    lookup = _return_product_lookup()
    raw["Product Name"] = raw["product_key"].map(lambda value: _return_product_name(value, lookup))
    raw["is_predicted_return"] = _return_flag(raw["predicted_return"])
    raw["return_probability"] = pd.to_numeric(raw["return_probability"], errors="coerce").fillna(0)
    filtered = _return_filters(raw)

    st.caption(f"Live analysis of {len(filtered):,} selected transactions from the current model output.")
    if filtered.empty:
        st.warning("No transactions match the selected filters. Adjust the filters to continue.")
        return

    predicted = filtered[filtered["is_predicted_return"]]
    total = len(filtered)
    predicted_count = len(predicted)
    rate = predicted_count / total
    city = predicted["customer_city"].value_counts().index[0] if not predicted.empty else "—"
    product = predicted["Product Name"].value_counts().index[0] if not predicted.empty else "—"

    kpis = st.columns(5)
    cards = [
        ("Total Transactions", f"{total:,}", "Current filtered population", "#8DA9C4", "#F0F5FA"),
        ("Predicted Returns", f"{predicted_count:,}", "Orders needing attention", "#B8A1C8", "#F6F1F8"),
        ("Return Rate", f"{rate:.1%}", "Predicted-return share", "#9BC5B0", "#F0F8F2"),
        ("Highest Return City", str(city).title(), "By predicted return volume", "#E7B98E", "#FCF6EE"),
        ("Most Returned Product", product, "By predicted return volume", "#A7C7C7", "#F0F8F8"),
    ]
    for col, card in zip(kpis, cards):
        with col:
            _return_kpi(*card)

    st.markdown("### Return risk overview")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        trend = filtered.groupby("month", as_index=False).agg(
            transactions=("transaction_id", "size"), predicted_returns=("is_predicted_return", "sum")
        ).sort_values("month")
        trend["month_label"] = trend["month"].map(lambda month: f"Month {int(month)}")
        trend_long = trend.melt("month_label", ["transactions", "predicted_returns"], "Metric", "Orders")
        fig = px.line(trend_long, x="month_label", y="Orders", color="Metric", markers=True,
                      color_discrete_sequence=["#8DA9C4", "#B8A1C8"], title="Predicted returns trend")
        st.plotly_chart(_return_chart_layout(fig), use_container_width=True)
    with chart_right:
        category = filtered.groupby("category", as_index=False).agg(
            transactions=("transaction_id", "size"), predicted_returns=("is_predicted_return", "sum")
        )
        category["return_rate"] = category["predicted_returns"] / category["transactions"]
        category = category.sort_values("return_rate").tail(10)
        fig = px.bar(category, x="return_rate", y="category", orientation="h", text="predicted_returns",
                     color="return_rate", color_continuous_scale=["#EAF2F1", "#7FA7A1"], title="Return risk by category",
                     labels={"return_rate": "Predicted return rate", "category": ""})
        fig.update_xaxes(tickformat=".0%")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(_return_chart_layout(fig), use_container_width=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        cities = filtered.groupby("customer_city", as_index=False).agg(
            transactions=("transaction_id", "size"), predicted_returns=("is_predicted_return", "sum")
        )
        cities["return_rate"] = cities["predicted_returns"] / cities["transactions"]
        cities = cities.sort_values("predicted_returns").tail(10)
        fig = px.bar(cities, x="predicted_returns", y="customer_city", orientation="h", text="return_rate",
                     color_discrete_sequence=["#8DA9C4"], title="Return risk by city",
                     labels={"predicted_returns": "Predicted returns", "customer_city": ""})
        fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
        st.plotly_chart(_return_chart_layout(fig), use_container_width=True)
    with chart_right:
        actual_reasons = filtered[
            filtered["normalized_return_reason"].notna()
            & ~filtered["normalized_return_reason"].astype(str).str.strip().str.lower().isin(["not returned", "no return", "none", "nan"])
        ]
        reasons = actual_reasons["normalized_return_reason"].value_counts().reset_index()
        reasons.columns = ["Return reason", "Orders"]
        if reasons.empty:
            st.info("No actual return reasons are available in this selection.")
        else:
            fig = px.pie(reasons, names="Return reason", values="Orders", hole=.62,
                         color_discrete_sequence=RETURN_CHART_COLORS, title="Actual return reason distribution")
            fig.update_traces(textinfo="percent+label", textposition="outside")
            st.plotly_chart(_return_chart_layout(fig), use_container_width=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig = px.histogram(filtered, x="return_probability", nbins=20, color="risk_level",
                           color_discrete_map={"Low": "#9BC5B0", "Medium": "#E7B98E", "High": "#D6A5A5"},
                           title="Prediction confidence distribution", labels={"return_probability": "Return probability", "count": "Transactions"})
        st.plotly_chart(_return_chart_layout(fig), use_container_width=True)
    with chart_right:
        products = predicted.groupby("Product Name", as_index=False).agg(predicted_returns=("transaction_id", "size")).nlargest(10, "predicted_returns")
        if products.empty:
            st.info("No predicted returns are available in this selection.")
        else:
            fig = px.bar(products.sort_values("predicted_returns"), x="predicted_returns", y="Product Name", orientation="h",
                         color_discrete_sequence=["#B8A1C8"], title="Top returned products",
                         labels={"predicted_returns": "Predicted returns", "Product Name": ""})
            st.plotly_chart(_return_chart_layout(fig), use_container_width=True)

    st.markdown("### Transaction detail")
    search = st.text_input("Search transactions", placeholder="Search product, city, company, category, order ID, or risk level")
    table = filtered.copy()
    if search:
        search_columns = ["transaction_id", "Product Name", "customer_city", "company_name", "category", "risk_level"]
        contains = table[search_columns].astype(str).apply(lambda column: column.str.contains(search, case=False, na=False))
        table = table[contains.any(axis=1)]
    table = table.sort_values(["return_probability", "transaction_id"], ascending=[False, True])
    table["Return Probability"] = table["return_probability"].map("{:.1%}".format)
    display = table[["transaction_id", "Product Name", "company_name", "category", "customer_city", "predicted_return", "risk_level", "Return Probability", "recommended_action"]].rename(columns={"transaction_id": "Transaction ID", "company_name": "Company", "category": "Category", "customer_city": "City", "predicted_return": "Predicted Return", "risk_level": "Risk Level", "recommended_action": "Recommended Action"})
    row_limit = st.select_slider("Rows per page", options=[25, 50, 100, 250], value=50)
    st.dataframe(display.head(row_limit).style.apply(_highlight_return_risk, axis=1), use_container_width=True, hide_index=True, column_config={
        "Risk Level": st.column_config.TextColumn("Risk Level"),
        "Return Probability": st.column_config.TextColumn("Return Probability"),
    })
    st.caption(f"Showing {min(len(display), row_limit):,} of {len(display):,} matching transactions. Sort columns directly in the table.")

    st.markdown("### Business Insights")
    category_rate = filtered.groupby("category")["is_predicted_return"].mean().sort_values(ascending=False)
    company_rate = filtered.groupby("company_name")["is_predicted_return"].mean().sort_values(ascending=False)
    avg_confidence = filtered["return_probability"].mean()
    insights = [
        f"{str(city).title()} contributes the most predicted returns in the current selection.",
        f"{str(category_rate.index[0]).title()} has the highest predicted return rate at {category_rate.iloc[0]:.1%}.",
        f"{str(company_rate.index[0]).title()} has the highest predicted return percentage at {company_rate.iloc[0]:.1%}.",
        f"{product} is the leading product driver, while the average model confidence is {avg_confidence:.1%}.",
    ]
    for insight in insights:
        st.markdown(f'<div class="return-insight">{insight}</div>', unsafe_allow_html=True)


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
    product_label = st.selectbox(
        "Product",
        options=products_df["product_id"],
        format_func=lambda pid: f"{pid} — {products_df.loc[products_df['product_id'] == pid, 'product_name'].values[0]}",
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
