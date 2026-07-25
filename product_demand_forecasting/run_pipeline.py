import subprocess

print("="*60)
print("PRODUCT DEMAND FORECASTING PIPELINE")
print("="*60)

print("\nStep 1: Extracting Data...")
subprocess.run(["python", "extract_data.py"], check=True)

print("\nStep 2: Feature Engineering...")
subprocess.run(["python", "feature_engineering.py"], check=True)

print("\nStep 3: Training Models...")
subprocess.run(["python", "train_model.py"], check=True)

print("\nStep 4: Generating Predictions...")
subprocess.run(["python", "predict.py"], check=True)

print("\nStep 5: Creating Visualizations...")
subprocess.run(["python", "visualization.py"], check=True)

print("\n✅ Product Demand Forecasting Pipeline Completed Successfully!")