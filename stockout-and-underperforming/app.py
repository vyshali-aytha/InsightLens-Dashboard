"""
app.py
------
REST API that runs the two database-backed analytics pipelines and exposes
every generated deliverable (JSON results, CSVs, graph images) to an
external dashboard, per the task's Phase 3 requirements.

Run:
    python app.py
Then, e.g.:
    POST /api/underperforming/run       -> executes the pipeline
    GET  /api/underperforming/results   -> full JSON results (cached)
    GET  /outputs/csv/<filename>        -> download a generated CSV
    GET  /outputs/graphs/<name>         -> fetch a generated graph PNG

Pipelines are NOT run automatically at startup (they hit the DB and can be
slow — ML training in particular) — call the /run endpoints to (re)compute,
then read from the cached /results, /csv, /graphs endpoints. Swap
`AUTO_RUN_ON_STARTUP = True` below if you'd rather they run once at boot.
"""

import os
import threading

from flask import Flask, jsonify, send_from_directory, abort

from pipelines import underperforming, stockout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "outputs", "csv")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

AUTO_RUN_ON_STARTUP = False

app = Flask(__name__)

# In-memory cache of the latest pipeline run for each system, plus a lock
# so two /run calls can't stomp on each other's output files concurrently.
_cache = {"underperforming": None, "stockout": None}
_locks = {"underperforming": threading.Lock(), "stockout": threading.Lock()}


def _run_underperforming():
    with _locks["underperforming"]:
        result = underperforming.run(CSV_DIR, GRAPHS_DIR)
        _cache["underperforming"] = result
        return result


def _run_stockout():
    with _locks["stockout"]:
        result = stockout.run(CSV_DIR, GRAPHS_DIR)
        _cache["stockout"] = result
        return result


def _require(system):
    result = _cache.get(system)
    if result is None:
        abort(409, description=f"No results yet for '{system}'. POST /api/{system}/run first.")
    return result


# =====================================================================
# Underperforming Product & City Detection
# =====================================================================
@app.route("/api/underperforming/run", methods=["POST"])
def run_underperforming():
    try:
        result = _run_underperforming()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({
        "status": "ok",
        "csv_files": sorted(result["csv_files"].keys()),
        "graph_files": sorted(result["graph_files"].keys()),
    })


@app.route("/api/underperforming/results", methods=["GET"])
def underperforming_results():
    result = _require("underperforming")
    return jsonify({k: v for k, v in result.items() if k not in ("csv_files", "graph_files")})


@app.route("/api/underperforming/products", methods=["GET"])
def underperforming_products():
    return jsonify(_require("underperforming")["product_recommendations"])


@app.route("/api/underperforming/cities", methods=["GET"])
def underperforming_cities():
    return jsonify(_require("underperforming")["city_recommendations"])


@app.route("/api/underperforming/top20", methods=["GET"])
def underperforming_top20():
    result = _require("underperforming")
    return jsonify({
        "top20_products": result["top20_products"],
        "top_cities": result["top_cities"],
    })


@app.route("/api/underperforming/statistics", methods=["GET"])
def underperforming_statistics():
    result = _require("underperforming")
    return jsonify({
        "method_agreement": result["method_agreement"],
        "flag_stability": result["flag_stability"],
    })


# =====================================================================
# Stockout & Reorder Prediction
# =====================================================================
@app.route("/api/stockout/run", methods=["POST"])
def run_stockout():
    try:
        result = _run_stockout()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({
        "status": "ok",
        "ml_ready": result["ml_ready"],
        "best_model": result["best_model"],
        "notes": result["notes"],
        "csv_files": sorted(result["csv_files"].keys()),
        "graph_files": sorted(result["graph_files"].keys()),
    })


@app.route("/api/stockout/results", methods=["GET"])
def stockout_results():
    result = _require("stockout")
    return jsonify({k: v for k, v in result.items() if k not in ("csv_files", "graph_files")})


@app.route("/api/stockout/predictions", methods=["GET"])
def stockout_predictions():
    return jsonify(_require("stockout")["recommendation_table"])


@app.route("/api/stockout/classification", methods=["GET"])
def stockout_classification():
    result = _require("stockout")
    return jsonify({
        "top20_highest_risk": result["top20_highest_risk"],
        "store_risk": result["store_risk"],
        "category_risk": result["category_risk"],
        "health_summary": result["health_summary"],
    })


@app.route("/api/stockout/metrics", methods=["GET"])
def stockout_metrics():
    result = _require("stockout")
    return jsonify({
        "ml_ready": result["ml_ready"],
        "best_model": result["best_model"],
        "model_evaluation_metrics": result["model_evaluation_metrics"],
        "rule_based_mae_days": result["rule_based_mae_days"],
    })


# =====================================================================
# Generic top-level aliases (per the spec's example endpoint list)
# =====================================================================
@app.route("/results", methods=["GET"])
def results():
    return jsonify({
        "underperforming": _cache["underperforming"] is not None,
        "stockout": _cache["stockout"] is not None,
    })


@app.route("/predictions", methods=["GET"])
def predictions():
    return stockout_predictions()


@app.route("/classification", methods=["GET"])
def classification():
    result = _require("underperforming")
    return jsonify({
        "product_severity_scores": result["product_severity_scores"],
        "city_severity_scores": result["city_severity_scores"],
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    return stockout_metrics()


@app.route("/statistics", methods=["GET"])
def statistics():
    return underperforming_statistics()


# =====================================================================
# File-serving endpoints — CSVs & graphs (dashboard never touches disk)
# =====================================================================
@app.route("/outputs/csv/<path:filename>", methods=["GET"])
def get_csv(filename):
    if not os.path.isfile(os.path.join(CSV_DIR, filename)):
        abort(404, description="No such CSV file.")
    return send_from_directory(CSV_DIR, filename, as_attachment=True)


@app.route("/outputs/graphs/<path:graph_name>", methods=["GET"])
def get_graph(graph_name):
    filename = graph_name if graph_name.endswith(".png") else f"{graph_name}.png"
    if not os.path.isfile(os.path.join(GRAPHS_DIR, filename)):
        abort(404, description="No such graph.")
    return send_from_directory(GRAPHS_DIR, filename, mimetype="image/png")


@app.route("/outputs/csv", methods=["GET"])
def list_csv():
    return jsonify(sorted(os.listdir(CSV_DIR)))


@app.route("/outputs/graphs", methods=["GET"])
def list_graphs():
    return jsonify(sorted(os.listdir(GRAPHS_DIR)))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    if AUTO_RUN_ON_STARTUP:
        try:
            _run_underperforming()
            _run_stockout()
        except Exception as e:
            print(f"Startup pipeline run failed (DB reachable?): {e}")
    app.run(host="0.0.0.0", port=5000, debug=False)
