# InsightLens Dashboard

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

- **Home** (`app.py`) — status card for every model.
- **Model Dashboard** (`pages/1_Model_Dashboard.py`) — pick a model from the
  sidebar dropdown, see its health, scores table, and charts.

## How data is fetched

Each model has a `type` in `config.py` that picks its render template and
loader functions, because the three models built so far each expose a
genuinely different API shape:

| type | model | API shape | CSV fallback folder |
|---|---|---|---|
| `risk_scoring` | Payment Failure Risk (Module 7) | `/health`, `/scores`, `/charts/<name>` | `data/payment_failure/` |
| `return_dashboard` | Return Risk Prediction | `/dashboard`, `/top-returned-products`, `/top-return-cities`, `/return-reasons`, `/high-risk-orders` | `data/return_risk/` |
| `discount_calculator` | Discount Impact Prediction | `POST /predict_discount` (live what-if, not a batch result) | `data/discount_impact/` (needs `dim_product.csv` for the product picker only) |

All three call the live API first (3s timeout). The first two fall back to
local CSVs if the API isn't running. The discount calculator is inherently
interactive (it prices a product+discount combo on request) so it has no
CSV fallback for the recommendation itself — only the product picklist
comes from a CSV.

CSV files expected in each folder:
- `data/payment_failure/`: `payment_failure_scores.csv` (from `payment_risk_scores.csv`), `payment_failure_scores_by_method_provider.csv`, `payment_failure_scores_by_month.csv`
- `data/return_risk/`: `predictions.csv`, `highly_returned_products.csv`, `top_return_cities.csv`, `return_reason_summary.csv` (straight from that model's `outputs/` folder)
- `data/discount_impact/`: `dim_product.csv`

## Adding the next model

1. Get its API running (or its CSVs sitting in `data/<model_key>/`).
2. Check its API shape. If it matches one of the three types above, add
   one entry to `MODELS` in `config.py` using that type as a template.
3. If it's a genuinely new shape, add a new `type`, one loader function in
   `utils/data_loader.py`, and one `render_*` function + dispatch branch in
   `pages/1_Model_Dashboard.py` — everything else (model picker, Home page
   cards) stays untouched.
