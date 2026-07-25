import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ===================================
# LOAD DATASET
# ===================================

print("Loading Dataset...")

df = pd.read_csv(
    "data/featured_sales_dataset.csv"
)

print("Dataset Shape:", df.shape)



# ===================================
# HANDLE CATEGORICAL COLUMNS
# ===================================

categorical_cols = [
    "product_name",
    "category",
    "subcategory",
    "store_name",
    "city",
    "state",
    "weekday"
]


encoders = {}


for col in categorical_cols:

    if col in df.columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = encoder



# ===================================
# FEATURES AND TARGET
# ===================================


features = [

    "product_key",
    "store_key",

    "product_name",
    "category",
    "subcategory",

    "store_name",
    "city",
    "state",

    "unit_price",
    "discount",

    "month",
    "quarter",
    "year",

    "weekday",

    "lag_1",
    "lag_3",
    "lag_7",

    "rolling_mean_3",
    "rolling_mean_7"
]


target = "quantity"



X = df[features].copy()

y = df[target]



# ===================================
# FINAL DATA TYPE CLEANING
# ===================================


for col in X.columns:

    if (
        X[col].dtype == "object"
        or str(X[col].dtype) == "string"
    ):

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )



print("\nFeature Datatypes:")
print(X.dtypes)



# ===================================
# TRAIN TEST SPLIT
# ===================================


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)



print("\nTraining Records :", len(X_train))

print("Testing Records :", len(X_test))





# ===================================
# PART 2
# MODEL TRAINING
# ===================================


# -----------------------------------
# XGBOOST
# -----------------------------------


print("\nTraining XGBoost Model...")


xgb_model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    random_state=42,

    objective="reg:squarederror"

)


xgb_model.fit(

    X_train,

    y_train

)


print("XGBoost Completed")





# -----------------------------------
# LIGHTGBM
# -----------------------------------


print("\nTraining LightGBM Model...")


lgbm_model = LGBMRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    random_state=42

)



lgbm_model.fit(

    X_train,

    y_train

)


print("LightGBM Completed")





# ===================================
# PREDICTIONS
# ===================================


xgb_pred = xgb_model.predict(
    X_test
)


lgbm_pred = lgbm_model.predict(
    X_test
)





# ===================================
# WMAPE FUNCTION
# ===================================


def calculate_wmape(actual, predicted):

    return (

        np.sum(
            np.abs(actual - predicted)
        )

        /

        np.sum(
            np.abs(actual)
        )

    ) * 100





# ===================================
# MODEL METRICS
# ===================================


xgb_mae = mean_absolute_error(

    y_test,

    xgb_pred

)


xgb_rmse = np.sqrt(

    mean_squared_error(

        y_test,

        xgb_pred

    )

)


xgb_wmape = calculate_wmape(

    y_test,

    xgb_pred

)





lgbm_mae = mean_absolute_error(

    y_test,

    lgbm_pred

)


lgbm_rmse = np.sqrt(

    mean_squared_error(

        y_test,

        lgbm_pred

    )

)


lgbm_wmape = calculate_wmape(

    y_test,

    lgbm_pred

)





# ===================================
# COMPARISON TABLE
# ===================================


results = pd.DataFrame({

    "Model":

    [

        "XGBoost",

        "LightGBM"

    ],


    "MAE":

    [

        xgb_mae,

        lgbm_mae

    ],


    "RMSE":

    [

        xgb_rmse,

        lgbm_rmse

    ],


    "WMAPE":

    [

        xgb_wmape,

        lgbm_wmape

    ]

})




print("\nModel Performance")

print(results)





# ===================================
# BEST MODEL
# ===================================


best_model_name = results.loc[

    results["WMAPE"].idxmin(),

    "Model"

]




if best_model_name == "XGBoost":

    best_model = xgb_model


else:

    best_model = lgbm_model




print(
    "\nBest Model Selected:",
    best_model_name
)
