"""
app.py
------
Streamlit web application for Delivery Time prediction.
Run locally with:  streamlit run app.py
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")


# ── Load model artifact ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. Please run `python train_model.py` first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


artifact = load_artifact()

pipeline        = artifact["pipeline"]
best_model_name = artifact["best_model_name"]
num_features    = artifact["numeric_features"]
cat_features    = artifact["categorical_features"]
metrics         = artifact["metrics"]
all_results     = artifact["all_results"]
cat_unique      = artifact["cat_unique"]
num_stats       = artifact["num_stats"]

# ── Helper: haversine distance ────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🚚",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🚚 Delivery Time Predictor")
st.markdown(
    "Enter the order and agent details below to get an estimated delivery time."
)

# ── Sidebar – model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Information")
    st.success(f"**Best Model:** {best_model_name}")
    st.metric("MAE (minutes)",  f"{metrics['MAE']:.2f}")
    st.metric("RMSE (minutes)", f"{metrics['RMSE']:.2f}")
    st.metric("R² Score",       f"{metrics['R2']:.4f}")

    st.subheader("All Models Compared")
    results_df = pd.DataFrame(all_results).T.rename(
        columns={"MAE": "MAE (min)", "RMSE": "RMSE (min)", "R2": "R²"}
    )
    results_df = results_df.sort_values("MAE (min)")
    st.dataframe(results_df.style.format({"MAE (min)": "{:.2f}", "RMSE (min)": "{:.2f}", "R²": "{:.4f}"}))

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📋 Order & Agent Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Agent Information**")
    agent_age    = st.slider(
        "Agent Age",
        min_value=int(num_stats["Agent_Age"]["min"]),
        max_value=int(num_stats["Agent_Age"]["max"]),
        value=int(num_stats["Agent_Age"]["mean"]),
    )
    agent_rating = st.slider(
        "Agent Rating",
        min_value=float(num_stats["Agent_Rating"]["min"]),
        max_value=float(num_stats["Agent_Rating"]["max"]),
        value=round(float(num_stats["Agent_Rating"]["mean"]), 1),
        step=0.1,
    )

with col2:
    st.markdown("**Order Conditions**")
    weather  = st.selectbox("Weather",  options=cat_unique["Weather"])
    traffic  = st.selectbox("Traffic",  options=cat_unique["Traffic"])
    vehicle  = st.selectbox("Vehicle",  options=cat_unique["Vehicle"])
    area     = st.selectbox("Area",     options=cat_unique["Area"])
    category = st.selectbox("Category", options=cat_unique["Category"])

with col3:
    st.markdown("**Location & Timing**")
    store_lat  = st.number_input("Store Latitude",  value=12.97, format="%.6f")
    store_lon  = st.number_input("Store Longitude", value=77.59, format="%.6f")
    drop_lat   = st.number_input("Drop Latitude",   value=13.04, format="%.6f")
    drop_lon   = st.number_input("Drop Longitude",  value=77.66, format="%.6f")

    order_time  = st.time_input("Order Time",  value=pd.Timestamp("2022-01-01 18:00").time())
    pickup_time = st.time_input("Pickup Time", value=pd.Timestamp("2022-01-01 18:12").time())
    order_date  = st.date_input("Order Date",  value=pd.Timestamp("2022-03-15"))

# ── Derived features ──────────────────────────────────────────────────────────
distance_km = haversine(store_lat, store_lon, drop_lat, drop_lon)

order_minutes  = order_time.hour  * 60 + order_time.minute
pickup_minutes = pickup_time.hour * 60 + pickup_time.minute
prep_time      = pickup_minutes - order_minutes
if prep_time < 0:
    prep_time += 1440   # midnight rollover

day_of_week = pd.Timestamp(order_date).dayofweek
hour_of_day = order_time.hour

# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("---")
predict_col, info_col = st.columns([1, 2])

with predict_col:
    predict_btn = st.button("🔮 Predict Delivery Time", use_container_width=True, type="primary")

with info_col:
    st.info(
        f"📏 Computed distance: **{distance_km:.2f} km**  |  "
        f"⏱ Prep time: **{prep_time} min**  |  "
        f"📅 Day of week: **{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}**"
    )

if predict_btn:
    input_df = pd.DataFrame(
        [[
            agent_age, agent_rating, distance_km,
            prep_time, day_of_week, hour_of_day,
            weather, traffic, vehicle, area, category,
        ]],
        columns=num_features + cat_features,
    )

    prediction = pipeline.predict(input_df)[0]
    prediction = max(0, round(float(prediction), 1))

    st.markdown("---")
    st.markdown("## 📦 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Estimated Delivery Time", f"{prediction:.1f} minutes")
    with res_col2:
        hours   = int(prediction) // 60
        minutes = int(prediction) % 60
        if hours > 0:
            st.metric("In Hours & Minutes", f"{hours}h {minutes}m")
        else:
            st.metric("In Hours & Minutes", f"{minutes}m")
    with res_col3:
        if prediction < 60:
            label, color = "⚡ Fast Delivery", "green"
        elif prediction < 120:
            label, color = "🕐 Standard Delivery", "orange"
        else:
            label, color = "🐢 Slow Delivery", "red"
        st.markdown(f"**Status:** :{color}[{label}]")

    with st.expander("📝 Input Summary"):
        summary = {
            "Agent Age": agent_age,
            "Agent Rating": agent_rating,
            "Distance (km)": f"{distance_km:.3f}",
            "Prep Time (min)": prep_time,
            "Weather": weather,
            "Traffic": traffic,
            "Vehicle": vehicle,
            "Area": area,
            "Category": category,
            "Order Hour": hour_of_day,
            "Day of Week": ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week],
        }
        st.table(pd.Series(summary, name="Value"))

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:12px;'>"
    "Delivery Time Predictor · Powered by scikit-learn &amp; Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
