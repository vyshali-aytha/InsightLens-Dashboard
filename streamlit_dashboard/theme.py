"""
InsightLens Dashboard — centralized visual theme.

Single source of truth for color. Anything that currently hardcodes a
hex value inline (status pills, card accents, chart colors) should pull
from here instead, so the palette can change in one place.

Pairs with .streamlit/config.toml, which sets the same palette for
Streamlit's own built-in widgets (buttons, sliders, selectbox, links).
"""

# Core palette — cool, data-forward, not the default warm-cream/terracotta
# or near-black/acid-green look. Deliberately muted rather than saturated:
# this is a working analytics tool, not a marketing page.
PALETTE = {
    "ink": "#1B2430",          # primary text
    "muted_text": "#6B7280",   # captions, secondary text
    "background": "#F7F8FA",  # page background
    "surface": "#FFFFFF",     # cards, containers
    "border": "#E2E5EA",      # hairlines, dividers, card borders
    "primary": "#2E6F6B",     # deep teal — primary accent, links, active states
}

# Status colors for model health (ok / degraded / offline). Muted rather
# than traffic-light-bright, so they read as "data status" not "alert".
STATUS_COLORS = {
    "ok":       {"fg": "#1F6B3A", "bg": "#E7F3EA"},
    "degraded": {"fg": "#8A6D00", "bg": "#FBF3D9"},
    "offline":  {"fg": "#A23B3B", "bg": "#F8E9E9"},
    "unknown":  {"fg": "#5B6472", "bg": "#EDEEF1"},
}

# One accent per model type, so a user learns to recognize "batch model"
# vs "live calculator" vs "KPI dashboard" at a glance without reading the
# label. Used as a left-border/accent stripe on each model card.
MODEL_TYPE_ACCENTS = {
    "risk_scoring": "#2E6F6B",        # teal — batch scoring service
    "return_dashboard": "#4A5FBA",    # indigo — KPI dashboard
    "discount_calculator": "#B5652E", # ochre — live what-if calculator
    "sales_forecast": "#3A8F6E",      # emerald — time-series forecasting
    "pipeline_dashboard": "#6B4F8C",  # violet — on-demand DB pipeline (POST /run)
}
