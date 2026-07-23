import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "discount_model.pkl"
SALES_FILE = "sales_new.csv"
OUTPUT_FILE = "recommendations.csv"

DISCOUNT_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

saved = joblib.load(MODEL_PATH)

model = saved["model"]

feature_columns = saved["feature_columns"]

print("=" * 70)
print("DISCOUNT RECOMMENDATION ENGINE")
print("=" * 70)

print("\nLoading sales data...")

sales = pd.read_csv(SALES_FILE)

print(f"Rows Loaded : {len(sales)}")

# ==========================================================
# CURRENT PRODUCT PERFORMANCE
# ==========================================================

current = (

    sales

    .groupby(

        [

            "product_id",

            "product_name",

            "category",

            "unit_price"

        ]

    )

    .agg(

        {

            "discount":"mean",

            "quantity":"mean",

            "gross_amount":"mean",

            "net_amount":"mean"

        }

    )

    .reset_index()

)

current.rename(

    columns={

        "discount":"current_discount",

        "quantity":"current_quantity",

        "gross_amount":"current_gross",

        "net_amount":"current_revenue"

    },

    inplace=True

)

print()

print("Products Found :",len(current))

# ==========================================================
# CREATE ALL POSSIBLE DISCOUNT SCENARIOS
# ==========================================================

rows=[]

for _,product in current.iterrows():

    for d in DISCOUNT_LEVELS:

        rows.append({

            "product_id":product.product_id,

            "product_name":product.product_name,

            "category":product.category,

            "unit_price":product.unit_price,

            "current_discount":product.current_discount,

            "current_quantity":product.current_quantity,

            "current_revenue":product.current_revenue,

            "discount":d

        })

scenario_df=pd.DataFrame(rows)

print()

print("Scenario Rows :",len(scenario_df))

# ==========================================================
# PREPARE FEATURES FOR MODEL
# ==========================================================

X=scenario_df[
    feature_columns
]

print()

print("Predicting Quantity...")

# ==========================================================
# BATCH PREDICTION
# ==========================================================

predicted_quantity = model.predict(X)

scenario_df["predicted_quantity"] = np.round(
    predicted_quantity
).astype(int)

scenario_df["predicted_quantity"] = scenario_df[
    "predicted_quantity"
].clip(lower=1)

# ==========================================================
# CALCULATE EXPECTED REVENUE
# ==========================================================

scenario_df["expected_gross"] = (

    scenario_df["predicted_quantity"]

    *

    scenario_df["unit_price"]

)

scenario_df["expected_revenue"] = (

    scenario_df["expected_gross"]

    *

    (1 - scenario_df["discount"]/100)

)

# ==========================================================
# FIND BEST DISCOUNT FOR EACH PRODUCT
# ==========================================================

best = (

    scenario_df

    .sort_values(

        "expected_revenue",

        ascending=False

    )

    .groupby("product_id")

    .head(1)

    .reset_index(drop=True)

)

print()

print("Best Discount Calculated For")

print(len(best),"Products")

# ==========================================================
# MERGE CURRENT + RECOMMENDED
# ==========================================================

recommendations = current.merge(

    best[

        [

            "product_id",

            "discount",

            "predicted_quantity",

            "expected_revenue"

        ]

    ],

    on="product_id"

)

recommendations.rename(

    columns={

        "discount":"recommended_discount",

        "predicted_quantity":"recommended_quantity",

        "expected_revenue":"recommended_revenue"

    },

    inplace=True

)

print()

print("Calculating Improvements...")

# ==========================================================
# REVENUE IMPROVEMENT
# ==========================================================

recommendations["revenue_gain"] = (

    recommendations["recommended_revenue"]

    -

    recommendations["current_revenue"]

)

recommendations["revenue_gain_percent"] = (

    recommendations["revenue_gain"]

    /

    recommendations["current_revenue"]

) * 100

# ==========================================================
# QUANTITY IMPROVEMENT
# ==========================================================

recommendations["quantity_gain"] = (

    recommendations["recommended_quantity"]

    -

    recommendations["current_quantity"]

)

recommendations["quantity_gain_percent"] = (

    recommendations["quantity_gain"]

    /

    recommendations["current_quantity"]

) * 100

# ==========================================================
# ROUND VALUES
# ==========================================================

recommendations["current_discount"] = recommendations[
    "current_discount"
].round(2)

recommendations["recommended_discount"] = recommendations[
    "recommended_discount"
].round(2)

recommendations["current_quantity"] = recommendations[
    "current_quantity"
].round(2)

recommendations["recommended_quantity"] = recommendations[
    "recommended_quantity"
].round(2)

recommendations["current_revenue"] = recommendations[
    "current_revenue"
].round(2)

recommendations["recommended_revenue"] = recommendations[
    "recommended_revenue"
].round(2)

recommendations["revenue_gain"] = recommendations[
    "revenue_gain"
].round(2)

recommendations["revenue_gain_percent"] = recommendations[
    "revenue_gain_percent"
].round(2)

recommendations["quantity_gain"] = recommendations[
    "quantity_gain"
].round(2)

recommendations["quantity_gain_percent"] = recommendations[
    "quantity_gain_percent"
].round(2)

# ==========================================================
# RECOMMENDED ACTION
# ==========================================================

def recommend_action(row):

    if row["recommended_discount"] > row["current_discount"]:

        return "Increase Discount"

    elif row["recommended_discount"] < row["current_discount"]:

        return "Reduce Discount"

    else:

        return "Keep Current Discount"

recommendations["action"] = recommendations.apply(

    recommend_action,

    axis=1

)

# ==========================================================
# PRIORITY LEVEL
# ==========================================================

def priority(row):

    gain = row["revenue_gain_percent"]

    if gain >= 20:

        return "High"

    elif gain >= 10:

        return "Medium"

    else:

        return "Low"

recommendations["priority"] = recommendations.apply(

    priority,

    axis=1

)

# ==========================================================
# SORT BY HIGHEST REVENUE GAIN
# ==========================================================

recommendations = recommendations.sort_values(

    "revenue_gain",

    ascending=False

).reset_index(drop=True)

print()

print("=" * 70)
print("Recommendation Summary")
print("=" * 70)

print()

print("Increase Discount :",

      (recommendations["action"]=="Increase Discount").sum())

print("Reduce Discount   :",

      (recommendations["action"]=="Reduce Discount").sum())

print("Keep Current      :",

      (recommendations["action"]=="Keep Current Discount").sum())

print()

print("High Priority     :",

      (recommendations["priority"]=="High").sum())

print("Medium Priority   :",

      (recommendations["priority"]=="Medium").sum())

print("Low Priority      :",

      (recommendations["priority"]=="Low").sum())

# ==========================================================
# OVERALL BUSINESS KPIs
# ==========================================================

total_current_revenue = recommendations["current_revenue"].sum()

total_expected_revenue = recommendations["recommended_revenue"].sum()

total_gain = recommendations["revenue_gain"].sum()

overall_gain_percent = (

    total_gain

    /

    total_current_revenue

) * 100

average_discount = recommendations["recommended_discount"].mean()

average_quantity = recommendations["recommended_quantity"].mean()

# ==========================================================
# EXPORT CSV
# ==========================================================

final_columns = [

    "product_id",

    "product_name",

    "category",

    "unit_price",

    "current_discount",

    "recommended_discount",

    "current_quantity",

    "recommended_quantity",

    "current_revenue",

    "recommended_revenue",

    "revenue_gain",

    "revenue_gain_percent",

    "quantity_gain",

    "quantity_gain_percent",

    "action",

    "priority"

]

recommendations = recommendations[final_columns]

recommendations.to_csv(

    OUTPUT_FILE,

    index=False

)

# ==========================================================
# PRINT TOP PRODUCTS
# ==========================================================

print()

print("=" * 80)
print("TOP 20 PRODUCTS WITH HIGHEST REVENUE IMPROVEMENT")
print("=" * 80)

print(

    recommendations.head(20).to_string(

        index=False

    )

)

# ==========================================================
# DASHBOARD KPIs
# ==========================================================

print()

print("=" * 80)
print("BUSINESS SUMMARY")
print("=" * 80)

print(f"Products Analysed           : {len(recommendations)}")

print(f"Current Revenue             : ₹{total_current_revenue:,.2f}")

print(f"Expected Revenue            : ₹{total_expected_revenue:,.2f}")

print(f"Expected Revenue Increase   : ₹{total_gain:,.2f}")

print(f"Revenue Growth              : {overall_gain_percent:.2f}%")

print(f"Average Recommended Discount: {average_discount:.2f}%")

print(f"Average Predicted Quantity  : {average_quantity:.2f}")

print()

print("=" * 80)

print("Recommendations saved as")

print(f"   {OUTPUT_FILE}")

print("=" * 80)

print()

print("Recommendation Engine Completed Successfully")