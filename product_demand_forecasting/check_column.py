import pandas as pd

df = pd.read_csv("data/featured_sales_dataset.csv")

print(df[[
    "product_name",
    "category",
    "subcategory",
    "store_name",
    "city",
    "state",
    "weekday"
]].head())