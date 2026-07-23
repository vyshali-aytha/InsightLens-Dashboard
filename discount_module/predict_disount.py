import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "discount_model.pkl"
PRODUCTS_FILE = "dim_product.csv"

DISCOUNT_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

# ==========================================================
# LOAD MODEL
# ==========================================================

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
feature_columns = saved_model["feature_columns"]

# ==========================================================
# LOAD PRODUCT DATA
# ==========================================================

products = pd.read_csv(PRODUCTS_FILE)

# Keep one row per product
products = (
    products[
        [
            "product_id",
            "product_name",
            "category",
            "unit_price"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def get_product(product_id):
    """
    Returns product details for the given product id.
    """

    result = products[
        products["product_id"].str.upper() == product_id.upper()
    ]

    if len(result) == 0:
        return None

    return result.iloc[0]


def predict_quantity(product, discount):
    """
    Predict quantity for a product at a given discount.
    """

    input_df = pd.DataFrame({

        "product_id": [product["product_id"]],

        "category": [product["category"]],

        "unit_price": [product["unit_price"]],

        "discount": [discount]

    })

    quantity = model.predict(input_df)[0]

    quantity = max(1, round(quantity))

    return quantity


def calculate_revenue(price, quantity, discount):

    gross = price * quantity

    revenue = gross * (1 - discount / 100)

    return round(revenue, 2)


def find_best_discount(product):
    """
    Test all discount levels and return the best one.
    """

    best_discount = None
    best_quantity = 0
    best_revenue = -1

    comparison = []

    for discount in DISCOUNT_LEVELS:

        qty = predict_quantity(product, discount)

        revenue = calculate_revenue(
            product["unit_price"],
            qty,
            discount
        )

        comparison.append({

            "discount": discount,

            "quantity": qty,

            "revenue": revenue

        })

        if revenue > best_revenue:

            best_revenue = revenue
            best_discount = discount
            best_quantity = qty

    comparison = pd.DataFrame(comparison)

    return (
        best_discount,
        best_quantity,
        best_revenue,
        comparison
    )

# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)
print("DISCOUNT IMPACT PREDICTION SYSTEM")
print("=" * 70)

# ==========================================================
# USER INPUT
# ==========================================================

print()

product_id = input("Enter Product ID : ").strip().upper()

product = get_product(product_id)

if product is None:

    print("\nProduct not found!")

    exit()

print()

while True:

    try:

        selected_discount = float(
            input("Enter Discount (0-40) : ")
        )

        if selected_discount < 0 or selected_discount > 40:

            print("Discount must be between 0 and 40.")

            continue

        break

    except ValueError:

        print("Please enter a valid number.")

# ==========================================================
# CURRENT SCENARIO PREDICTION
# ==========================================================

predicted_quantity = predict_quantity(

    product,

    selected_discount

)

expected_revenue = calculate_revenue(

    product["unit_price"],

    predicted_quantity,

    selected_discount

)

# ==========================================================
# MODEL RECOMMENDATION
# ==========================================================

(

    recommended_discount,

    recommended_quantity,

    recommended_revenue,

    comparison_df

) = find_best_discount(product)

# ==========================================================
# CALCULATE IMPROVEMENT
# ==========================================================

revenue_gain = (

    recommended_revenue

    -

    expected_revenue

)

if expected_revenue > 0:

    revenue_growth = (

        revenue_gain

        /

        expected_revenue

    ) * 100

else:

    revenue_growth = 0

quantity_gain = (

    recommended_quantity

    -

    predicted_quantity

)

# ==========================================================
# RECOMMENDED ACTION
# ==========================================================

if recommended_discount > selected_discount:

    action = "Increase Discount"

elif recommended_discount < selected_discount:

    action = "Reduce Discount"

else:

    action = "Current Discount is Optimal"

# ==========================================================
# DISPLAY RESULT
# ==========================================================

print()

print("=" * 75)
print("DISCOUNT IMPACT PREDICTION RESULT")
print("=" * 75)

print(f"Product ID          : {product['product_id']}")
print(f"Product Name        : {product['product_name']}")
print(f"Category            : {product['category']}")
print(f"Unit Price          : ₹{product['unit_price']:,.2f}")

print()

print("-" * 75)
print("CURRENT SCENARIO")
print("-" * 75)

print(f"Selected Discount   : {selected_discount:.0f}%")
print(f"Predicted Quantity  : {predicted_quantity}")
print(f"Expected Revenue    : ₹{expected_revenue:,.2f}")

print()

print("-" * 75)
print("MODEL RECOMMENDATION")
print("-" * 75)

print(f"Recommended Discount : {recommended_discount}%")
print(f"Predicted Quantity   : {recommended_quantity}")
print(f"Expected Revenue     : ₹{recommended_revenue:,.2f}")

print()

print("-" * 75)
print("IMPROVEMENT")
print("-" * 75)

print(f"Revenue Gain         : ₹{revenue_gain:,.2f}")
print(f"Revenue Growth       : {revenue_growth:.2f}%")
print(f"Quantity Increase    : {quantity_gain}")

print()

print(f"Recommended Action   : {action}")

print("=" * 75)

# ==========================================================
# DISCOUNT COMPARISON TABLE
# ==========================================================

print()
print("=" * 75)
print("DISCOUNT ANALYSIS")
print("=" * 75)

comparison_display = comparison_df.copy()

comparison_display.rename(
    columns={
        "discount": "Discount (%)",
        "quantity": "Predicted Quantity",
        "revenue": "Expected Revenue (₹)"
    },
    inplace=True
)

print(comparison_display.to_string(index=False))

# ==========================================================
# HIGHLIGHT BEST DISCOUNT
# ==========================================================

best_row = comparison_df.loc[
    comparison_df["revenue"].idxmax()
]

print()
print("=" * 75)
print("BEST DISCOUNT SUMMARY")
print("=" * 75)

print(f"Best Discount        : {int(best_row['discount'])}%")
print(f"Predicted Quantity   : {int(best_row['quantity'])}")
print(f"Expected Revenue     : ₹{best_row['revenue']:,.2f}")

# ==========================================================
# SAVE ANALYSIS
# ==========================================================

comparison_output = comparison_df.copy()

comparison_output.insert(
    0,
    "product_id",
    product["product_id"]
)

comparison_output.insert(
    1,
    "product_name",
    product["product_name"]
)

comparison_output.insert(
    2,
    "category",
    product["category"]
)

comparison_output.insert(
    3,
    "unit_price",
    product["unit_price"]
)

comparison_output.to_csv(
    f"{product['product_id']}_discount_analysis.csv",
    index=False
)

print()
print(f"Detailed analysis saved as '{product['product_id']}_discount_analysis.csv'")

# ==========================================================
# BUSINESS INSIGHT
# ==========================================================

print()
print("=" * 75)
print("BUSINESS INSIGHT")
print("=" * 75)

if revenue_gain > 0:

    print(
        f"""
Increasing the discount from {selected_discount:.0f}% to
{recommended_discount}% is expected to increase revenue by
₹{revenue_gain:,.2f} ({revenue_growth:.2f}%).

Expected sales volume increases from
{predicted_quantity} units to
{recommended_quantity} units.
"""
    )

elif revenue_gain < 0:

    print(
        f"""
The selected discount already performs better than
the model recommendation.

No increase in revenue is expected.
"""
    )

else:

    print(
        """
Current discount already appears to be optimal.
No change is recommended.
"""
    )

print("=" * 75)

print()
print("Prediction Completed Successfully.")