from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Discount Prediction API",
    description="API to recommend the optimal discount for a given product.",
    version="1.0.0"
)

MODEL_PATH = "discount_model.pkl"
PRODUCTS_FILE = "dim_product.csv"
DISCOUNT_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

# Load model and product data on startup
try:
    saved_model = joblib.load(MODEL_PATH)
    model = saved_model["model"]
    feature_columns = saved_model.get("feature_columns", [])
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

try:
    products = pd.read_csv(PRODUCTS_FILE)
    products = products[["product_id", "product_name", "category", "unit_price"]].drop_duplicates().reset_index(drop=True)
except Exception as e:
    products = None
    print(f"Error loading products: {e}")

class PredictionRequest(BaseModel):
    product_id: str
    discount: float

class DiscountCurvePoint(BaseModel):
    discount: float
    predicted_quantity: int
    expected_revenue: float

class PredictionResponse(BaseModel):
    product_id: str
    product_name: str
    category: str
    unit_price: float
    current_discount: float
    current_predicted_quantity: int
    current_expected_revenue: float
    recommended_discount: float
    recommended_predicted_quantity: int
    recommended_expected_revenue: float
    revenue_gain: float
    recommended_action: str
    discount_curve: list[DiscountCurvePoint]

def get_product(product_id: str):
    if products is None:
        return None
    result = products[products["product_id"].str.upper() == product_id.upper()]
    if len(result) == 0:
        return None
    return result.iloc[0]

def predict_quantity(product, discount: float):
    input_df = pd.DataFrame({
        "product_id": [product["product_id"]],
        "category": [product["category"]],
        "unit_price": [product["unit_price"]],
        "discount": [discount]
    })
    quantity = model.predict(input_df)[0]
    return max(1, round(quantity))

def calculate_revenue(price: float, quantity: int, discount: float):
    gross = price * quantity
    revenue = gross * (1 - discount / 100)
    return round(revenue, 2)

@app.post("/predict_discount", response_model=PredictionResponse)
def predict_discount(request: PredictionRequest):
    if model is None or products is None:
        raise HTTPException(status_code=500, detail="Model or product data not loaded.")
        
    product = get_product(request.product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found.")
        
    if request.discount < 0 or request.discount > 100:
        raise HTTPException(status_code=400, detail="Discount must be between 0 and 100.")
        
    # Predict for the current scenario
    current_qty = predict_quantity(product, request.discount)
    current_revenue = calculate_revenue(product["unit_price"], current_qty, request.discount)
    
    # Predict best discount level
    best_discount = None
    best_quantity = 0
    best_revenue = -1
    
    discount_curve = []
    
    for discount in DISCOUNT_LEVELS:
        qty = predict_quantity(product, discount)
        revenue = calculate_revenue(product["unit_price"], qty, discount)
        
        discount_curve.append(DiscountCurvePoint(
            discount=float(discount),
            predicted_quantity=int(qty),
            expected_revenue=float(revenue)
        ))
        
        if revenue > best_revenue:
            best_revenue = revenue
            best_discount = discount
            best_quantity = qty
            
    # Recommended Action
    if best_discount > request.discount:
        action = "Increase Discount"
    elif best_discount < request.discount:
        action = "Reduce Discount"
    else:
        action = "Current Discount is Optimal"
        
    revenue_gain = best_revenue - current_revenue
        
    return PredictionResponse(
        product_id=product["product_id"],
        product_name=product["product_name"],
        category=product["category"],
        unit_price=float(product["unit_price"]),
        current_discount=request.discount,
        current_predicted_quantity=int(current_qty),
        current_expected_revenue=float(current_revenue),
        recommended_discount=float(best_discount),
        recommended_predicted_quantity=int(best_quantity),
        recommended_expected_revenue=float(best_revenue),
        revenue_gain=float(revenue_gain),
        recommended_action=action,
        discount_curve=discount_curve
    )
