import random
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TOTAL_ROWS = 15000

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)

# --------------------------------------------------
# Load Products
# --------------------------------------------------

products = pd.read_csv("dim_product.csv")

# --------------------------------------------------
# Discount probabilities
# --------------------------------------------------

discount_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40]

discount_weights = [
    35,
    22,
    15,
    10,
    8,
    5,
    3,
    1,
    1
]

# --------------------------------------------------
# Category popularity
# --------------------------------------------------

category_factor = {

    "Electronics":0.8,
    "Grocery":2.2,
    "Sports":1.2,
    "Home Appliances":0.9,
    "Apparel":1.6,
    "Health":1.5,
    "Automotive":0.8,
    "Stationery":1.4,
    "Pet Care":1.2,
    "Garden":0.9,
    "Beauty":1.6,
    "Books":1.3,
    "Toys":1.5,
    "Footwear":1.4,
    "Furniture":0.6

}

# --------------------------------------------------
# Assign fixed popularity + elasticity
# --------------------------------------------------

product_profile = {}

for _, row in products.iterrows():

    product_profile[row.product_id] = {

        "popularity": random.uniform(0.8, 2.5),

        "elasticity": random.uniform(0.7, 1.8)

    }

# --------------------------------------------------
# Weekend factor
# --------------------------------------------------

def weekend_factor(date):

    if date.weekday() >= 5:
        return 1.20

    return 1.00

# --------------------------------------------------
# Seasonal factor
# --------------------------------------------------

def seasonal_factor(date):

    month = date.month

    if month in [10,11,12]:
        return 1.25

    if month in [6,7]:
        return 0.90

    return 1.00

# --------------------------------------------------
# Discount response
# --------------------------------------------------

def discount_multiplier(discount, elasticity):

    x = discount / 10

    return 1 + elasticity * (1 - math.exp(-0.55 * x))

# --------------------------------------------------
# Generate
# --------------------------------------------------

rows = []

for i in range(TOTAL_ROWS):

    product = products.sample(1).iloc[0]

    profile = product_profile[product.product_id]

    discount = random.choices(

        discount_levels,

        weights=discount_weights,

        k=1

    )[0]

    date = START_DATE + timedelta(

        days=random.randint(0,365)

    )

    base_demand = profile["popularity"] * 6

    quantity = (

        base_demand

        * category_factor[product.category]

        * weekend_factor(date)

        * seasonal_factor(date)

        * discount_multiplier(

            discount,

            profile["elasticity"]

        )

    )

    quantity += np.random.normal(0,1.3)

    quantity = max(1, round(quantity))

    price = float(product.unit_price)

    gross = quantity * price

    net = gross * (1-discount/100)

    rows.append({

        "transaction_id":f"T{i+1:06}",

        "product_id":product.product_id,

        "product_name":product.product_name,

        "category":product.category,

        "sale_date":date.date(),

        "unit_price":price,

        "discount":discount,

        "quantity":quantity,

        "gross_amount":round(gross,2),

        "net_amount":round(net,2)

    })

sales = pd.DataFrame(rows)

sales.to_csv(

    "sales_new.csv",

    index=False

)

print()

print("="*60)

print("Rows Generated :",len(sales))

print()

print(sales.head())

print("="*60)

print()

print("Average Quantity by Discount")

print(

    sales.groupby("discount")["quantity"].mean()

)
