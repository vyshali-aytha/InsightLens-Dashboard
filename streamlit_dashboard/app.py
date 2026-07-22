import streamlit as st
from config import MODELS
from utils.data_loader import get_health
from theme import PALETTE, STATUS_COLORS, MODEL_TYPE_ACCENTS

st.set_page_config(page_title="InsightLens", layout="wide")

CARD_CSS = f"""
<style>
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 10px;
    border-color: {PALETTE["border"]} !important;
    transition: box-shadow 0.15s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: 0 2px 10px rgba(27, 36, 48, 0.08);
}}
.pipeline-step {{
    text-align: center;
    padding: 14px 8px;
    border-radius: 10px;
    background: {PALETTE["surface"]};
    border: 1px solid {PALETTE["border"]};
    height: 100%;
}}
.pipeline-step .step-title {{ font-weight: 600; color: {PALETTE["ink"]}; }}
.pipeline-step .step-sub {{ font-size: 0.8rem; color: {PALETTE["muted_text"]}; margin-top: 2px; }}
.pipeline-arrow {{ text-align: center; font-size: 1.4rem; color: {PALETTE["border"]}; padding-top: 22px; }}
.status-pill {{
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600;
}}
.model-accent {{
    height: 4px; border-radius: 4px; margin-bottom: 10px;
}}
</style>
"""


def home():
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.title("InsightLens")
    st.caption("Retail analytics ETL pipeline — model dashboards")

    # --- Pipeline overview: where these models sit in the bigger picture ---
    st.divider()
    st.subheader("Pipeline")
    active_count = sum(1 for v in MODELS.values() if v["status"] == "active")
    steps = [
        ("Stage 1", "Validate raw MART_*.csv files"),
        ("Stage 2", "Business rules (BR-01–BR-22)"),
        ("Stage 3", "Load star-schema warehouse"),
        ("Models", f"{active_count} scoring/prediction services"),
        ("Dashboard", "This app"),
    ]
    cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
    for i, (title, sub) in enumerate(steps):
        with cols[i * 2]:
            st.markdown(
                f'<div class="pipeline-step">'
                f'<div class="step-title">{title}</div><div class="step-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )
        if i < len(steps) - 1:
            with cols[i * 2 + 1]:
                st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

    # --- Model cards ---
    st.divider()
    st.subheader("Models")
    active_models = {k: v for k, v in MODELS.items() if v["status"] == "active"}
    cols = st.columns(3)

    for i, (model_key, cfg) in enumerate(active_models.items()):
        col = cols[i % 3]
        with col:
            with st.container(border=True):
                accent = MODEL_TYPE_ACCENTS.get(cfg.get("type"), PALETTE["border"])
                st.markdown(f'<div class="model-accent" style="background:{accent};"></div>', unsafe_allow_html=True)
                st.markdown(f"### {cfg['label']}")

                if cfg.get("type") == "risk_scoring":
                    health = get_health(model_key, cfg["api_base"])
                    status = health.get("status", "unknown")
                    colors = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
                    st.markdown(
                        f'<span class="status-pill" style="background:{colors["bg"]};color:{colors["fg"]};">{status}</span>',
                        unsafe_allow_html=True,
                    )
                    if health.get("rows_scored") is not None:
                        st.metric("Rows scored", health["rows_scored"])
                    if health.get("last_scored_at"):
                        st.caption(f"Last refreshed: {health['last_scored_at']}")
                else:
                    # return_dashboard and discount_calculator have no /health
                    # endpoint — just point to the Model Dashboard page
                    # instead of guessing at a status.
                    st.caption("See Model Dashboard for details →")

    st.divider()
    st.info("Open **Model Dashboard** in the sidebar to explore a specific model's scores and charts.")


pg = st.navigation([
    st.Page(home, title="Home", url_path="home", default=True),
    st.Page("pages/1_Model_Dashboard.py", title="Model Dashboard", url_path="model-dashboard"),
])
pg.run()
