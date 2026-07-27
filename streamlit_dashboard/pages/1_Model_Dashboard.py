import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config import MODELS
from theme import STATUS_COLORS, PALETTE
from utils.data_loader import (
    get_health, get_scores, get_chart_data,
    get_return_dashboard, get_return_breakdown, get_return_predictions,
    get_discount_products, post_discount_prediction,
    get_sales_forecast_data, get_sales_historical_data,
    get_demand_forecast_data, get_demand_model_metrics,
    post_pipeline_run, get_pipeline_json, pipeline_graph_url, get_pipeline_csv_bytes,
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
            
            if "discount_curve" in result:
                curve_df = pd.DataFrame(result["discount_curve"])
                curve_df["discount"] = curve_df["discount"].astype(int)
                
                st.markdown("### Expected Revenue & Quantity by Discount Level")
                import altair as alt
                
                base = alt.Chart(curve_df).encode(
                    x=alt.X('discount:O', title='Discount (%)')
                )
                
                bar = base.mark_bar(opacity=0.6, color="#B8A1C8").encode(
                    y=alt.Y('predicted_quantity:Q', title='Predicted Quantity'),
                    tooltip=['discount', 'predicted_quantity', 'expected_revenue']
                )
                
                line = base.mark_line(color="#2E5C55", point=True).encode(
                    y=alt.Y('expected_revenue:Q', title='Expected Revenue', axis=alt.Axis(format="$.2f")),
                    tooltip=['discount', 'predicted_quantity', 'expected_revenue']
                )
                
                current_rule = alt.Chart(pd.DataFrame({'discount': [int(result['current_discount'])]})).mark_rule(
                    color='gray', strokeDash=[4, 4], size=2
                ).encode(x='discount:O')
                
                recommended_rule = alt.Chart(pd.DataFrame({'discount': [int(result['recommended_discount'])]})).mark_rule(
                    color='#0b5fa5', strokeDash=[2, 2], size=2
                ).encode(x='discount:O')
                
                chart = alt.layer(bar, line, current_rule, recommended_rule).resolve_scale(y='independent').properties(height=400)
                st.altair_chart(chart, use_container_width=True)
                
                st.caption("Gray dashed line = Current discount. Blue dashed line = Recommended discount.")

            with st.expander("Full response"):
                st.json(result)


# =============================================================================
# type: sales_forecast — Sales & Revenue Forecasting dashboard
# =============================================================================

SALES_FORECAST_CSS = """
<style>
.sf-kpi {
    min-height: 122px; padding: 16px 18px; border: 1px solid var(--accent);
    border-left: 5px solid var(--accent); border-radius: 14px;
    background: var(--tint); box-shadow: 0 5px 16px rgba(20, 31, 48, .05);
    transition: transform .16s ease, box-shadow .16s ease;
}
.sf-kpi:hover { transform: translateY(-2px); box-shadow: 0 9px 22px rgba(20, 31, 48, .09); }
.sf-kpi__label { color: #52606D; font-size: .76rem; font-weight: 700; letter-spacing: .045em; text-transform: uppercase; }
.sf-kpi__value { color: #1B2430; font-size: 1.6rem; font-weight: 750; line-height: 1.35; overflow-wrap: anywhere; }
.sf-kpi__sub { color: #6B7280; font-size: .82rem; }
.sf-insight { padding: 14px 16px; background: #FFFFFF; border: 1px solid #E2E5EA; border-radius: 10px; margin-bottom: 10px; color: #394150; }
</style>
"""

SALES_CHART_COLORS = ["#3A8F6E", "#5BB58E", "#2E6F6B", "#8DA9C4", "#B8A1C8", "#E7B98E"]


def _sf_kpi(label, value, subtitle, accent, tint):
    st.markdown(
        f'<div class="sf-kpi" style="--accent:{accent};--tint:{tint};">'
        f'<div class="sf-kpi__label">{label}</div>'
        f'<div class="sf-kpi__value">{value}</div>'
        f'<div class="sf-kpi__sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def _sf_chart_layout(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="white",
        plot_bgcolor="white", font=dict(color="#1B2430"), legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EDF0F3", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def _sf_filters(df):
    """Store/product/date filters for the forecast data."""
    filtered = df.copy()
    with st.expander("Filters", expanded=False):
        st.caption("Refine the forecast output. All dashboard elements update automatically.")
        row = st.columns(3)
        with row[0]:
            stores = sorted(filtered["store_name"].dropna().astype(str).unique())
            sel_stores = st.multiselect("Store", stores, key="sf_store")
            if sel_stores:
                filtered = filtered[filtered["store_name"].astype(str).isin(sel_stores)]
        with row[1]:
            products = sorted(filtered["product_name"].dropna().astype(str).unique())
            sel_products = st.multiselect("Product", products, key="sf_product")
            if sel_products:
                filtered = filtered[filtered["product_name"].astype(str).isin(sel_products)]
        with row[2]:
            months = sorted(filtered["month"].dropna().unique())
            month_labels = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            month_options = [int(m) for m in months]
            sel_months = st.multiselect(
                "Month", month_options,
                format_func=lambda m: month_labels.get(m, str(m)),
                key="sf_month",
            )
            if sel_months:
                filtered = filtered[filtered["month"].isin(sel_months)]
    return filtered


def render_sales_forecast(cfg):
    forecast = get_sales_forecast_data(cfg)
    if forecast is None or forecast.empty:
        st.warning("No forecast data available — check the API is running or place forecast_results_1.csv in the model's data folder.")
        return

    st.markdown(SALES_FORECAST_CSS, unsafe_allow_html=True)

    # Ensure types
    forecast["full_date"] = pd.to_datetime(forecast["full_date"])
    for col in ["Actual Revenue", "Predicted Revenue", "total_orders", "avg_order_value"]:
        if col in forecast.columns:
            forecast[col] = pd.to_numeric(forecast[col], errors="coerce")
    for col in ["store_id", "product_id", "month", "year", "quarter"]:
        if col in forecast.columns:
            forecast[col] = pd.to_numeric(forecast[col], errors="coerce")

    # Apply filters
    filtered = _sf_filters(forecast)
    if filtered.empty:
        st.warning("No transactions match the selected filters. Adjust the filters to continue.")
        return

    st.caption(f"Analysing {len(filtered):,} forecast records.")

    # ---- KPI Cards ----
    total_predicted = filtered["Predicted Revenue"].sum()
    total_actual = filtered["Actual Revenue"].sum()
    mape = float(np.mean(np.abs(
        (filtered["Actual Revenue"] - filtered["Predicted Revenue"]) / filtered["Actual Revenue"]
    )) * 100)
    accuracy = 100 - mape
    date_min = filtered["full_date"].min().strftime("%b %d, %Y")
    date_max = filtered["full_date"].max().strftime("%b %d, %Y")
    top_store = (
        filtered.groupby("store_name")["Predicted Revenue"].sum()
        .sort_values(ascending=False).index[0]
    ) if not filtered.empty else "—"
    top_product = (
        filtered.groupby("product_name")["Predicted Revenue"].sum()
        .sort_values(ascending=False).index[0]
    ) if not filtered.empty else "—"

    kpis = st.columns(5)
    cards = [
        ("Predicted Revenue", f"₹{total_predicted:,.0f}", "Total forecasted revenue", "#3A8F6E", "#EEF8F3"),
        ("Actual Revenue", f"₹{total_actual:,.0f}", "Total actual revenue", "#2E6F6B", "#EDF5F4"),
        ("Model Accuracy", f"{accuracy:.1f}%", f"MAPE: {mape:.1f}%", "#5BB58E", "#F0FAF4"),
        ("Top Store", str(top_store), "By predicted revenue", "#8DA9C4", "#F0F5FA"),
        ("Top Product", str(top_product), "By predicted revenue", "#B8A1C8", "#F6F1F8"),
    ]
    for col, card in zip(kpis, cards):
        with col:
            _sf_kpi(*card)

    # ---- Charts Row 1: Trend + Monthly Summary ----
    st.markdown("### Revenue Forecast Analysis")
    chart_left, chart_right = st.columns(2)

    with chart_left:
        daily = filtered.sort_values("full_date").copy()
        daily["Date"] = daily["full_date"].dt.strftime("%Y-%m-%d")
        trend_long = daily.melt(
            id_vars=["Date"],
            value_vars=["Actual Revenue", "Predicted Revenue"],
            var_name="Metric", value_name="Revenue",
        )
        fig = px.line(
            trend_long, x="Date", y="Revenue", color="Metric", markers=True,
            color_discrete_sequence=["#3A8F6E", "#8DA9C4"],
            title="Actual vs Predicted Revenue Trend",
        )
        st.plotly_chart(_sf_chart_layout(fig), use_container_width=True)

    with chart_right:
        month_labels = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        monthly = (
            filtered.groupby("month", as_index=False)
            .agg({"Actual Revenue": "sum", "Predicted Revenue": "sum"})
            .sort_values("month")
        )
        monthly["Month"] = monthly["month"].map(month_labels)
        monthly_long = monthly.melt(
            id_vars=["Month", "month"],
            value_vars=["Actual Revenue", "Predicted Revenue"],
            var_name="Metric", value_name="Revenue",
        ).sort_values("month")
        fig = px.bar(
            monthly_long, x="Month", y="Revenue", color="Metric", barmode="group",
            color_discrete_sequence=["#3A8F6E", "#8DA9C4"],
            title="Monthly Revenue: Actual vs Predicted",
        )
        st.plotly_chart(_sf_chart_layout(fig), use_container_width=True)

    # ---- Charts Row 2: By Store + By Product ----
    chart_left, chart_right = st.columns(2)

    with chart_left:
        store_rev = (
            filtered.groupby("store_name", as_index=False)
            .agg({"Predicted Revenue": "sum", "Actual Revenue": "sum"})
            .sort_values("Predicted Revenue")
        )
        fig = px.bar(
            store_rev, x="Predicted Revenue", y="store_name", orientation="h",
            color_discrete_sequence=["#3A8F6E"],
            title="Forecasted Revenue by Store",
            labels={"Predicted Revenue": "Predicted Revenue (₹)", "store_name": ""},
        )
        st.plotly_chart(_sf_chart_layout(fig), use_container_width=True)

    with chart_right:
        product_rev = (
            filtered.groupby("product_name", as_index=False)
            .agg({"Predicted Revenue": "sum", "Actual Revenue": "sum"})
            .sort_values("Predicted Revenue")
            .tail(15)
        )
        fig = px.bar(
            product_rev, x="Predicted Revenue", y="product_name", orientation="h",
            color_discrete_sequence=["#5BB58E"],
            title="Forecasted Revenue by Product (Top 15)",
            labels={"Predicted Revenue": "Predicted Revenue (₹)", "product_name": ""},
        )
        st.plotly_chart(_sf_chart_layout(fig), use_container_width=True)

    # ---- Forecast Accuracy Section ----
    st.markdown("### Forecast Accuracy")
    chart_left, chart_right = st.columns(2)

    with chart_left:
        max_val = max(filtered["Actual Revenue"].max(), filtered["Predicted Revenue"].max()) * 1.05
        fig = px.scatter(
            filtered,
            x="Actual Revenue", y="Predicted Revenue",
            color="store_name",
            color_discrete_sequence=SALES_CHART_COLORS,
            title="Actual vs Predicted (Calibration)",
            labels={"Actual Revenue": "Actual Revenue (₹)", "Predicted Revenue": "Predicted Revenue (₹)"},
            hover_data=["store_name", "product_name", "full_date"],
        )
        # 45° reference line
        fig.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="gray", dash="dash", width=1),
        )
        st.plotly_chart(_sf_chart_layout(fig, height=400), use_container_width=True)
        st.caption("Points on the dashed line = perfect prediction. Deviation = error.")

    with chart_right:
        filtered_copy = filtered.copy()
        filtered_copy["Residual"] = filtered_copy["Actual Revenue"] - filtered_copy["Predicted Revenue"]
        fig = px.histogram(
            filtered_copy, x="Residual", nbins=30,
            color_discrete_sequence=["#3A8F6E"],
            title="Residual Distribution (Actual − Predicted)",
            labels={"Residual": "Residual (₹)", "count": "Frequency"},
        )
        st.plotly_chart(_sf_chart_layout(fig, height=400), use_container_width=True)
        st.caption("A tight cluster around zero indicates consistent predictions.")

    # ---- Transaction Detail Table ----
    st.markdown("### Forecast Detail")
    search = st.text_input(
        "Search forecasts",
        placeholder="Search store, product, or date",
        key="sf_search",
    )
    table = filtered.copy()
    table["Error"] = table["Actual Revenue"] - table["Predicted Revenue"]
    table["Error %"] = ((table["Error"].abs() / table["Actual Revenue"]) * 100).round(1)
    table["Date"] = table["full_date"].dt.strftime("%Y-%m-%d")

    if search:
        search_cols = ["Date", "store_name", "product_name"]
        contains = table[search_cols].astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        )
        table = table[contains.any(axis=1)]

    table = table.sort_values("full_date", ascending=False)
    display = table[[
        "Date", "store_name", "product_name", "total_orders",
        "Actual Revenue", "Predicted Revenue", "Error", "Error %",
    ]].rename(columns={
        "store_name": "Store",
        "product_name": "Product",
        "total_orders": "Orders",
    })

    row_limit = st.select_slider(
        "Rows per page", options=[25, 50, 100, 250], value=50, key="sf_rows",
    )
    st.dataframe(
        display.head(row_limit),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Actual Revenue": st.column_config.NumberColumn("Actual Revenue", format="₹%.0f"),
            "Predicted Revenue": st.column_config.NumberColumn("Predicted Revenue", format="₹%.0f"),
            "Error": st.column_config.NumberColumn("Error", format="₹%.0f"),
            "Error %": st.column_config.NumberColumn("Error %", format="%.1f%%"),
        },
    )
    st.caption(f"Showing {min(len(display), row_limit):,} of {len(display):,} records.")

    # ---- Business Insights ----
    st.markdown("### Business Insights")
    store_rev_total = filtered.groupby("store_name")["Predicted Revenue"].sum().sort_values(ascending=False)
    product_rev_total = filtered.groupby("product_name")["Predicted Revenue"].sum().sort_values(ascending=False)
    total_orders = filtered["total_orders"].sum()
    avg_order = filtered["avg_order_value"].mean()

    insights = [
        f"**{store_rev_total.index[0]}** leads in predicted revenue with ₹{store_rev_total.iloc[0]:,.0f} across the forecast period.",
        f"**{product_rev_total.index[0]}** is the top forecasted product, contributing ₹{product_rev_total.iloc[0]:,.0f} in predicted sales.",
        f"The model achieves **{accuracy:.1f}% accuracy** (MAPE: {mape:.1f}%) across {len(filtered):,} forecast records.",
        f"Total forecasted orders: **{total_orders:,.0f}** with an average order value of **₹{avg_order:,.0f}**.",
        f"Forecast period: **{date_min}** to **{date_max}** ({len(filtered['month'].unique())} months).",
    ]
    for insight in insights:
        st.markdown(f'<div class="sf-insight">{insight}</div>', unsafe_allow_html=True)


# =============================================================================
# type: demand_forecast — Product Demand Forecasting dashboard
# (product_demand_forecasting's api.py shape — /health, /forecast,
# /forecast/summary. Quantity per product/store, not revenue.)
# =============================================================================

DF_KPI_CSS = """
<style>
.df-kpi {
    min-height: 122px; padding: 16px 18px; border: 1px solid var(--accent);
    border-left: 5px solid var(--accent); border-radius: 14px;
    background: var(--tint); box-shadow: 0 5px 16px rgba(20, 31, 48, .05);
    transition: transform .16s ease, box-shadow .16s ease;
}
.df-kpi:hover { transform: translateY(-2px); box-shadow: 0 9px 22px rgba(20, 31, 48, .09); }
.df-kpi__label { color: #52606D; font-size: .76rem; font-weight: 700; letter-spacing: .045em; text-transform: uppercase; }
.df-kpi__value { color: #1B2430; font-size: 1.6rem; font-weight: 750; line-height: 1.35; overflow-wrap: anywhere; }
.df-kpi__sub { color: #6B7280; font-size: .82rem; }
.df-insight { padding: 14px 16px; background: #FFFFFF; border: 1px solid #E2E5EA; border-radius: 10px; margin-bottom: 10px; color: #394150; }
</style>
"""

DEMAND_CHART_COLORS = ["#C9752E", "#E0A868", "#8A5626", "#8DA9C4", "#B8A1C8", "#9BC5B0"]


def _df_kpi(label, value, subtitle, accent, tint):
    st.markdown(
        f'<div class="df-kpi" style="--accent:{accent};--tint:{tint};">'
        f'<div class="df-kpi__label">{label}</div>'
        f'<div class="df-kpi__value">{value}</div>'
        f'<div class="df-kpi__sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def _df_chart_layout(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="white",
        plot_bgcolor="white", font=dict(color="#1B2430"), legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EDF0F3", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def _df_filters(df):
    """Store/product/category filters for the demand forecast data."""
    filtered = df.copy()
    with st.expander("Filters", expanded=False):
        st.caption("Refine the forecast output. All dashboard elements update automatically.")
        row = st.columns(3)
        with row[0]:
            stores = sorted(filtered["store_name"].dropna().astype(str).unique())
            sel_stores = st.multiselect("Store", stores, key="df_store")
            if sel_stores:
                filtered = filtered[filtered["store_name"].astype(str).isin(sel_stores)]
        with row[1]:
            products = sorted(filtered["product_name"].dropna().astype(str).unique())
            sel_products = st.multiselect("Product", products, key="df_product")
            if sel_products:
                filtered = filtered[filtered["product_name"].astype(str).isin(sel_products)]
        with row[2]:
            if "category" in filtered.columns:
                categories = sorted(filtered["category"].dropna().astype(str).unique())
                sel_categories = st.multiselect("Category", categories, key="df_category")
                if sel_categories:
                    filtered = filtered[filtered["category"].astype(str).isin(sel_categories)]
    return filtered


def render_product_demand_forecast(cfg):
    forecast = get_demand_forecast_data(cfg)
    if forecast is None or forecast.empty:
        st.warning("No forecast data available — check the API is running or place future_demand_forecast.csv in the model's data folder.")
        return

    st.markdown(DF_KPI_CSS, unsafe_allow_html=True)

    forecast = forecast.copy()
    if "full_date" in forecast.columns:
        forecast["full_date"] = pd.to_datetime(forecast["full_date"])
    for col in ["quantity", "predicted_quantity", "unit_price", "discount"]:
        if col in forecast.columns:
            forecast[col] = pd.to_numeric(forecast[col], errors="coerce")

    filtered = _df_filters(forecast)
    if filtered.empty:
        st.warning("No records match the selected filters. Adjust the filters to continue.")
        return

    st.caption(f"Analysing {len(filtered):,} demand forecast records.")

    # ---- KPI Cards ----
    total_actual_qty = filtered["quantity"].sum()
    total_predicted_qty = filtered["predicted_quantity"].sum()
    nonzero = filtered[filtered["quantity"] != 0]
    wmape = (
        (nonzero["quantity"] - nonzero["predicted_quantity"]).abs().sum()
        / nonzero["quantity"].abs().sum() * 100
    ) if not nonzero.empty else float("nan")
    top_product = (
        filtered.groupby("product_name")["predicted_quantity"].sum()
        .sort_values(ascending=False).index[0]
    ) if not filtered.empty else "—"
    top_store = (
        filtered.groupby("store_name")["predicted_quantity"].sum()
        .sort_values(ascending=False).index[0]
    ) if not filtered.empty else "—"

    kpis = st.columns(5)
    cards = [
        ("Actual Demand", f"{total_actual_qty:,.0f} units", "Total actual quantity", "#C9752E", "#FCF3EA"),
        ("Forecasted Demand", f"{total_predicted_qty:,.0f} units", "Total predicted quantity", "#8A5626", "#F7EDE3"),
        ("Forecast Error", f"{wmape:.1f}% WMAPE", "Weighted mean abs. % error", "#E0A868", "#FDF5EA"),
        ("Top Product", str(top_product), "By predicted demand", "#8DA9C4", "#F0F5FA"),
        ("Top Store", str(top_store), "By predicted demand", "#B8A1C8", "#F6F1F8"),
    ]
    for col, card in zip(kpis, cards):
        with col:
            _df_kpi(*card)

    # ---- Charts Row 1: Trend + By Category ----
    st.markdown("### Demand Forecast Analysis")
    chart_left, chart_right = st.columns(2)

    with chart_left:
        daily = filtered.groupby(filtered["full_date"].dt.date, as_index=False).agg(
            {"quantity": "sum", "predicted_quantity": "sum"}
        ).rename(columns={"full_date": "Date"})
        daily.columns = ["Date", "Actual Quantity", "Predicted Quantity"]
        trend_long = daily.melt("Date", ["Actual Quantity", "Predicted Quantity"], "Metric", "Quantity")
        fig = px.line(trend_long, x="Date", y="Quantity", color="Metric", markers=False,
                      color_discrete_sequence=["#C9752E", "#8DA9C4"], title="Actual vs Predicted Demand Trend")
        st.plotly_chart(_df_chart_layout(fig), use_container_width=True)

    with chart_right:
        if "category" in filtered.columns:
            cat = filtered.groupby("category", as_index=False).agg(
                {"quantity": "sum", "predicted_quantity": "sum"}
            ).sort_values("predicted_quantity").tail(10)
            fig = px.bar(cat, x="predicted_quantity", y="category", orientation="h",
                         color_discrete_sequence=["#C9752E"], title="Forecasted Demand by Category",
                         labels={"predicted_quantity": "Predicted Quantity", "category": ""})
            st.plotly_chart(_df_chart_layout(fig), use_container_width=True)
        else:
            st.info("No category column available in this data.")

    # ---- Charts Row 2: By Store + By Product ----
    chart_left, chart_right = st.columns(2)

    with chart_left:
        store_qty = (
            filtered.groupby("store_name", as_index=False)
            .agg({"predicted_quantity": "sum", "quantity": "sum"})
            .sort_values("predicted_quantity").tail(15)
        )
        fig = px.bar(store_qty, x="predicted_quantity", y="store_name", orientation="h",
                     color_discrete_sequence=["#8A5626"], title="Forecasted Demand by Store",
                     labels={"predicted_quantity": "Predicted Quantity", "store_name": ""})
        st.plotly_chart(_df_chart_layout(fig), use_container_width=True)

    with chart_right:
        product_qty = (
            filtered.groupby("product_name", as_index=False)
            .agg({"predicted_quantity": "sum", "quantity": "sum"})
            .sort_values("predicted_quantity").tail(15)
        )
        fig = px.bar(product_qty, x="predicted_quantity", y="product_name", orientation="h",
                     color_discrete_sequence=["#E0A868"], title="Forecasted Demand by Product (Top 15)",
                     labels={"predicted_quantity": "Predicted Quantity", "product_name": ""})
        st.plotly_chart(_df_chart_layout(fig), use_container_width=True)

    # ---- Forecast Accuracy ----
    st.markdown("### Forecast Accuracy")
    chart_left, chart_right = st.columns(2)

    with chart_left:
        max_val = max(filtered["quantity"].max(), filtered["predicted_quantity"].max()) * 1.05
        fig = px.scatter(
            filtered.sample(min(len(filtered), 3000), random_state=42),
            x="quantity", y="predicted_quantity", color="store_name",
            color_discrete_sequence=DEMAND_CHART_COLORS,
            title="Actual vs Predicted (Calibration)",
            labels={"quantity": "Actual Quantity", "predicted_quantity": "Predicted Quantity"},
            hover_data=["product_name", "store_name"],
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                     line=dict(color="gray", dash="dash", width=1))
        st.plotly_chart(_df_chart_layout(fig, height=400), use_container_width=True)
        st.caption("Points on the dashed line = perfect prediction. Deviation = error. Sampled for readability.")

    with chart_right:
        residuals = filtered.copy()
        residuals["Residual"] = residuals["quantity"] - residuals["predicted_quantity"]
        fig = px.histogram(residuals, x="Residual", nbins=30, color_discrete_sequence=["#C9752E"],
                           title="Residual Distribution (Actual − Predicted)",
                           labels={"Residual": "Residual (units)", "count": "Frequency"})
        st.plotly_chart(_df_chart_layout(fig, height=400), use_container_width=True)
        st.caption("A tight cluster around zero indicates consistent predictions.")

    # ---- Model Comparison ----
    metrics_df = get_demand_model_metrics(cfg)
    if metrics_df is not None and not metrics_df.empty:
        st.markdown("### Model Comparison")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption("WMAPE (weighted mean absolute % error) selects the model used for the live forecast — lower is better.")

    # ---- Detail Table ----
    st.markdown("### Forecast Detail")
    search = st.text_input("Search forecasts", placeholder="Search store, product, or category", key="df_search")
    table = filtered.copy()
    table["Error"] = table["quantity"] - table["predicted_quantity"]
    table["Date"] = table["full_date"].dt.strftime("%Y-%m-%d")

    if search:
        search_cols = [c for c in ["Date", "store_name", "product_name", "category"] if c in table.columns]
        contains = table[search_cols].astype(str).apply(lambda col: col.str.contains(search, case=False, na=False))
        table = table[contains.any(axis=1)]

    table = table.sort_values("full_date", ascending=False)
    display_cols = ["Date", "store_name", "product_name", "category", "quantity", "predicted_quantity", "Error"]
    display_cols = [c for c in display_cols if c in table.columns]
    display = table[display_cols].rename(columns={
        "store_name": "Store", "product_name": "Product", "category": "Category",
        "quantity": "Actual Qty", "predicted_quantity": "Predicted Qty",
    })
    row_limit = st.select_slider("Rows per page", options=[25, 50, 100, 250], value=50, key="df_rows")
    st.dataframe(
        display.head(row_limit), use_container_width=True, hide_index=True,
        column_config={
            "Predicted Qty": st.column_config.NumberColumn("Predicted Qty", format="%.1f"),
            "Error": st.column_config.NumberColumn("Error", format="%.1f"),
        },
    )
    st.caption(f"Showing {min(len(display), row_limit):,} of {len(display):,} records.")

    # ---- Business Insights ----
    st.markdown("### Business Insights")
    product_total = filtered.groupby("product_name")["predicted_quantity"].sum().sort_values(ascending=False)
    store_total = filtered.groupby("store_name")["predicted_quantity"].sum().sort_values(ascending=False)
    insights = [
        f"**{product_total.index[0]}** has the highest forecasted demand at {product_total.iloc[0]:,.0f} units.",
        f"**{store_total.index[0]}** is expected to see the most demand, at {store_total.iloc[0]:,.0f} predicted units.",
        f"The model achieves a **{wmape:.1f}% WMAPE** across {len(filtered):,} forecast records.",
        f"Total forecasted demand: **{total_predicted_qty:,.0f} units** vs **{total_actual_qty:,.0f} units** actual.",
    ]
    for insight in insights:
        st.markdown(f'<div class="df-insight">{insight}</div>', unsafe_allow_html=True)


# =============================================================================
# type: pipeline_dashboard — Flask API shape (project/app.py):
# POST /run executes the DB-backed pipeline; GET endpoints read the
# server-side cache. No CSV fallback — the Flask API is the only source.
# =============================================================================
def render_pipeline_dashboard(model_key, cfg):
    st.caption("Runs a database-backed pipeline via the Flask API. Results stay cached until you run it again.")
    if st.button(f"Run {cfg['label']}", type="primary", key=f"run_{model_key}"):
        with st.spinner("Running pipeline against the warehouse — this can take a while..."):
            run_result = post_pipeline_run(cfg)
        if not run_result or "error" in run_result:
            st.error(f"Pipeline run failed: {(run_result or {}).get('error', 'unknown error')}")
        else:
            st.success("Pipeline run complete.")
            get_pipeline_json.clear()  # server cache changed — drop our stale GET cache
            get_pipeline_csv_bytes.clear()

    endpoints = cfg["endpoints"]
    data_keys = [k for k in endpoints if k not in ("run", "csv_list", "csv_file", "graph_list", "graph_file")]
    fetched = {k: get_pipeline_json(model_key, cfg, k) for k in data_keys}

    if all(v is None for v in fetched.values()):
        st.info("No results yet for this run — click **Run** above (POST /.../run) to compute them.")
        return

    tab_labels = [k.replace("_", " ").title() for k in data_keys] + ["Graphs", "CSV downloads"]
    tabs = st.tabs(tab_labels)

    for tab, key in zip(tabs, data_keys):
        with tab:
            value = fetched[key]
            if value is None:
                st.info("No data returned for this endpoint yet.")
            elif isinstance(value, list):
                st.dataframe(pd.DataFrame(value), use_container_width=True)
            elif isinstance(value, dict):
                # dicts of sub-tables (e.g. statistics -> method_agreement/flag_stability,
                # classification -> top20_highest_risk/store_risk/...)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        st.markdown(f"**{sub_key.replace('_', ' ').title()}**")
                        st.dataframe(pd.DataFrame(sub_value), use_container_width=True)
                    else:
                        st.metric(sub_key.replace("_", " ").title(), sub_value)

    with tabs[-2]:
        available_graphs = get_pipeline_json(model_key, cfg, "graph_list") or []
        graph_names = cfg.get("graphs", [])
        shown = False
        for name in graph_names:
            if f"{name}.png" in available_graphs:
                st.image(pipeline_graph_url(cfg, name), caption=name.replace("_", " ").title(), use_container_width=True)
                shown = True
        if not shown:
            st.info("No graphs available yet — run the pipeline first.")

    with tabs[-1]:
        available_csvs = get_pipeline_json(model_key, cfg, "csv_list") or []
        if not available_csvs:
            st.info("No CSVs available yet — run the pipeline first.")
        for filename in available_csvs:
            csv_bytes = get_pipeline_csv_bytes(cfg, filename)
            if csv_bytes:
                st.download_button(f"Download {filename}", data=csv_bytes, file_name=filename, mime="text/csv", key=f"dl_{model_key}_{filename}")


# --- Dispatch on type ---
model_type = cfg.get("type")
if model_type == "risk_scoring":
    render_risk_scoring(model_key, cfg)
elif model_type == "return_dashboard":
    render_return_dashboard(cfg)
elif model_type == "discount_calculator":
    render_discount_calculator(cfg)
elif model_type == "sales_forecast":
    render_sales_forecast(cfg)
elif model_type == "demand_forecast":
    render_product_demand_forecast(cfg)
elif model_type == "pipeline_dashboard":
    render_pipeline_dashboard(model_key, cfg)
else:
    st.info("This model doesn't have a render type configured yet.")
