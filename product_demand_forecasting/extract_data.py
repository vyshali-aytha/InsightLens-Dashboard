"""
Module 2: Product Demand Forecasting

Extracts sales and related data from the PostgreSQL database
for model training and forecasting.
"""

import psycopg2
import pandas as pd

# Database Connection
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="bbi",      
    user="postgres",
    password="vanshika"  # password
)

query = """
SELECT
    fs.transaction_id,
    dd.full_date,
    dd.day,
    dd.month,
    dd.quarter,
    dd.year,
    dd.weekday,

    fs.product_key,
    dp.product_name,
    dp.category,
    dp.subcategory,
    dp.unit_price,

    fs.store_key,
    ds.store_name,
    ds.city,
    ds.state,

    fs.quantity,
    fs.discount,
    fs.gross_amount,
    fs.net_amount

FROM insightlens.fact_sales fs
JOIN insightlens.dim_date dd
ON fs.date_key = dd.date_key

JOIN insightlens.dim_product dp
ON fs.product_key = dp.product_key

JOIN insightlens.dim_store ds
ON fs.store_key = ds.store_key;
"""

df = pd.read_sql(query, conn)

print(df.head())

df.to_csv("data/sales_dataset.csv", index=False)

print("Dataset Saved Successfully!")

conn.close()