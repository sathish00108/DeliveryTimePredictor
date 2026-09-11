"""
train_model.py
--------------
Trains ML models on the delivery-time dataset, compares them, and saves
the best pipeline to models/model.pkl.

Dataset: data/dataset.csv
Target : Delivery_Time (minutes) — regression problem
"""

import os
import math
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ── Strip whitespace from string columns ─────────────────────────────────────
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

# ── Feature engineering ───────────────────────────────────────────────────────
# 1. Haversine distance between store and drop location
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

df["Distance_km"] = haversine(
    df["Store_Latitude"], df["Store_Longitude"],
    df["Drop_Latitude"],  df["Drop_Longitude"],
)

# 2. Preparation time: minutes between order and pickup
def time_to_minutes(t):
    """Convert HH:MM:SS or HH:MM string to total minutes since midnight.
    Returns NaN for missing/unparseable values."""
    if pd.isna(t):
        return np.nan
    parts = str(t).split(":")
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return np.nan

df["Order_Minutes"]  = df["Order_Time"].apply(time_to_minutes)
df["Pickup_Minutes"] = df["Pickup_Time"].apply(time_to_minutes)
df["Prep_Time_min"]  = (df["Pickup_Minutes"] - df["Order_Minutes"]).apply(
    lambda x: x if x >= 0 else x + 1440   # handle midnight rollover
)

# 3. Day of week and hour from order date/time
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Day_of_Week"] = df["Order_Date"].dt.dayofweek   # 0=Mon … 6=Sun
df["Hour_of_Day"] = df["Order_Minutes"] // 60

# ── Define features ───────────────────────────────────────────────────────────
TARGET = "Delivery_Time"

DROP_COLS = [
    "Order_ID", "Order_Date", "Order_Time", "Pickup_Time",
    "Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude",
    "Order_Minutes", "Pickup_Minutes",
    TARGET,
]

NUMERIC_FEATURES = [
    "Agent_Age", "Agent_Rating", "Distance_km",
    "Prep_Time_min", "Day_of_Week", "Hour_of_Day",
]
CATEGORICAL_FEATURES = ["Weather", "Traffic", "Vehicle", "Area", "Category"]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = df[TARGET].astype(float)

# Remove rows where target is NaN
mask = y.notna()
X, y = X[mask], y[mask]

print(f"  Usable samples after cleaning: {len(X)}")

# ── Preprocessing pipelines ───────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,      NUMERIC_FEATURES),
    ("cat", categorical_transformer,  CATEGORICAL_FEATURES),
])

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Model candidates ──────────────────────────────────────────────────────────
candidates = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest":    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Extra Trees":      ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    ),
}

results = {}
print("\nTraining and evaluating models ...")
for name, model in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    results[name] = {"pipeline": pipe, "MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"  {name:<25}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

# ── Select best model (lowest MAE) ───────────────────────────────────────────
best_name = min(results, key=lambda k: results[k]["MAE"])
best_result = results[best_name]
print(f"\n>> Best model: {best_name}  (MAE={best_result['MAE']:.2f}, R2={best_result['R2']:.4f})")

# ── Save pipeline + metadata ──────────────────────────────────────────────────
artifact = {
    "pipeline":             best_result["pipeline"],
    "best_model_name":      best_name,
    "numeric_features":     NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "target":               TARGET,
    "metrics": {
        "MAE":  best_result["MAE"],
        "RMSE": best_result["RMSE"],
        "R2":   best_result["R2"],
    },
    "all_results": {
        k: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"]}
        for k, v in results.items()
    },
    # Store unique values for categorical dropdowns in app.py
    "cat_unique": {
        col: sorted(df[col].dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES
    },
    "num_stats": {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in NUMERIC_FEATURES
    },
}

joblib.dump(artifact, MODEL_PATH)
print(f">> Model saved: {MODEL_PATH}")
