"""
InsightLens Dashboard — model registry.

To add a new model once it's ready: add one entry to MODELS below.
Nothing else in the app needs to change — the Home page and the
Model Dashboard page both read this dict.

Fields common to every model:
    label        : display name shown in the UI
    icon         : currently unused (emoji removed) — reserved for a future
                   proper icon set on the Home page card
    type         : which render template pages/1_Model_Dashboard.py uses.
                   One of "risk_scoring", "return_dashboard", "discount_calculator".
                   Add a new type + matching loader/render code only when a
                   genuinely new API shape shows up — don't force a model
                   into a type it doesn't match.
    api_base     : base URL of the model's FastAPI service (None if not built yet)
    csv_dir      : local folder to fall back to if the API is unreachable
    status       : "active" | "coming_soon"

Remaining fields depend on `type` — see the comment above each block below.
"""

MODELS = {
    # type: risk_scoring — batch scorer exposing /health, /scores, /charts/*
    # (module7_api.py's shape). endpoints.charts keys must match both the
    # API's /charts/<key> and the local CSV suffix _scores_<key>.csv.
    "payment_failure": {
        "label": "Payment Failure Risk",
        "icon": None,
        "type": "risk_scoring",
        "api_base": "http://localhost:8007",
        "csv_dir": "data/payment_failure",
        "endpoints": {
            "health": "/health",
            "scores": "/scores",
            "charts": {
                "by_method_provider": "/charts/by_method_provider",
                "by_month": "/charts/by_month",
                "calibration": "/charts/calibration",
            },
        },
        "status": "active",
    },

    # type: return_dashboard — api.py's shape: /dashboard (summary KPIs) +
    # a handful of /top-*, /return-reasons, /high-risk-orders endpoints,
    # each backed by its own CSV in outputs/. No /health endpoint, so the
    # dashboard page treats "dashboard endpoint reachable" as the health check.
    "return_risk": {
        "label": "Return Risk Prediction",
        "icon": None,
        "type": "return_dashboard",
        "api_base": "http://localhost:8009",
        "csv_dir": "data/return_risk",
        "endpoints": {
            "dashboard": "/dashboard",
            "top_returned_products": "/top-returned-products",
            "top_return_cities": "/top-return-cities",
            "return_reasons": "/return-reasons",
            "high_risk_orders": "/high-risk-orders",
        },
        # local CSV fallback filenames, matching outputs/ in the model's own project
        "csv_files": {
            "predictions": "predictions.csv",
            "top_returned_products": "highly_returned_products.csv",
            "top_return_cities": "top_return_cities.csv",
            "return_reasons": "return_reason_summary.csv",
        },
        "status": "active",
    },

    # type: discount_calculator — not a batch model: POST /predict_discount
    # with a product_id + discount, get back a recommended discount /
    # revenue estimate. The dashboard renders this as a pick-a-product tool,
    # not a table + charts. Product list comes from dim_product.csv (the
    # API itself has no "list products" endpoint).
    "discount_impact": {
        "label": "Discount Impact Prediction",
        "icon": None,
        "type": "discount_calculator",
        "api_base": "http://localhost:8010",
        "csv_dir": "data/discount_impact",
        "endpoints": {
            "predict": "/predict_discount",
        },
        "products_csv": "dim_product.csv",
        "discount_levels": [0, 5, 10, 15, 20, 25, 30, 35, 40],
        "status": "active",
    },

    # type: sales_forecast — time-series revenue forecasting using a
    # RandomForest trained on sales_ml_input.  Exposes /health, /forecast,
    # /historical, /forecast/summary.  CSV fallback uses the model's own
    # forecast_results_1.csv + sales_ml_input.csv.
    "sales_forecast": {
        "label": "Sales & Revenue Forecasting",
        "icon": None,
        "type": "sales_forecast",
        "api_base": "http://localhost:8011",
        "csv_dir": "data/sales_forecast",
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "historical": "/historical",
            "summary": "/forecast/summary",
        },
        "csv_files": {
            "forecast": "forecast_results_1.csv",
            "historical": "sales_ml_input.csv",
        },
        "status": "active",
    },

    # type: demand_forecast — per-product/per-store quantity forecasting
    # (product_demand_forecasting's api.py shape, mirrors sales_forecast).
    # Exposes /health, /forecast, /forecast/summary. CSV fallback uses the
    # model's own future_demand_forecast.csv (already scored with
    # predicted_quantity) + model_metrics.csv.
    "product_demand": {
        "label": "Product Demand Forecasting",
        "icon": None,
        "type": "demand_forecast",
        "api_base": "http://localhost:8012",
        "csv_dir": "data/product_demand",
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "summary": "/forecast/summary",
        },
        "csv_files": {
            "forecast": "future_demand_forecast.csv",
            "metrics": "model_metrics.csv",
        },
        "status": "active",
    },

    # type: pipeline_dashboard — Flask API shape (project/app.py): pipelines
    # are NOT precomputed. POST .../run executes the DB-backed pipeline and
    # caches the result server-side; GET endpoints below just read that
    # cache (no DB hit, no recompute). No CSV fallback — this is the
    # Dashboard -> HTTP -> Flask API -> Warehouse flow, so the Flask API is
    # the only data source. csv_file/graph_file are downloaded/rendered
    # straight from the API's own file-serving routes.
    "underperforming": {
        "label": "Underperforming Product & City Detection",
        "icon": None,
        "type": "pipeline_dashboard",
        "api_base": "http://localhost:5000",
        "endpoints": {
            "run": "/api/underperforming/run",
            "products": "/api/underperforming/products",
            "cities": "/api/underperforming/cities",
            "statistics": "/api/underperforming/statistics",
            "csv_list": "/outputs/csv",
            "csv_file": "/outputs/csv/{filename}",
            "graph_list": "/outputs/graphs",
            "graph_file": "/outputs/graphs/{name}",
        },
        # graphs this page renders, if present in the API's graph_list
        "graphs": ["product_dashboard", "city_dashboard", "severity_distribution", "trend_zscore_distribution"],
        "status": "active",
    },

    "stockout": {
        "label": "Stockout & Reorder Prediction",
        "icon": None,
        "type": "pipeline_dashboard",
        "api_base": "http://localhost:5000",
        "endpoints": {
            "run": "/api/stockout/run",
            "predictions": "/api/stockout/predictions",
            "classification": "/api/stockout/classification",
            "metrics": "/api/stockout/metrics",
            "csv_list": "/outputs/csv",
            "csv_file": "/outputs/csv/{filename}",
            "graph_list": "/outputs/graphs",
            "graph_file": "/outputs/graphs/{name}",
        },
        "graphs": ["inventory_status_distribution", "feature_importance"],
        "status": "active",
    },
}
