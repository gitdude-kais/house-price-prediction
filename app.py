import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction (Linear Regression)",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.subheader("Linear Regression Model")

# -----------------------------
# Load trained artifacts
# -----------------------------
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------
# Load dataset to get city names
# -----------------------------
data = pd.read_csv("processed_house_data.csv")
cities = sorted(data["city"].unique())

# -----------------------------
# User Inputs
# -----------------------------
st.markdown("### 🏡 Enter House Details")

bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, step=0.5)
sqft_living = st.number_input("Sqft Living Area", min_value=100, step=50)
sqft_lot = st.number_input("Sqft Lot Area", min_value=500, step=100)
floors = st.number_input("Floors", min_value=1.0, step=0.5)

waterfront = st.selectbox("Waterfront", [0, 1])
view = st.selectbox("View Rating", [0, 1, 2, 3, 4])
condition = st.selectbox("Condition", [1, 2, 3, 4, 5])

sqft_above = st.number_input("Sqft Above", min_value=100, step=50)
sqft_basement = st.number_input("Sqft Basement", min_value=0, step=50)

month = st.selectbox("Month", list(range(1, 13)))
day = st.selectbox("Day", list(range(1, 32)))

is_renovated = st.selectbox("Renovated", [0, 1])

city = st.selectbox("City", cities)

# -----------------------------
# Build input dictionary
# -----------------------------
input_data = {
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "waterfront": waterfront,
    "view": view,
    "condition": condition,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,
    "month": month,
    "day": day,
    "is_renovated": is_renovated
}

# One-hot encode city
for feature in feature_names:
    if feature.startswith("city_"):
        input_data[feature] = 1 if feature == f"city_{city}" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])

    # Ensure correct column order
    input_df = input_df[feature_names]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    raw_prediction = model.predict(input_scaled)[0]

    # 🔒 SAFETY FIX: prevent negative prices
    final_prediction = max(raw_prediction, 0)

    st.success(f"💰 Predicted House Price: ${final_prediction:,.2f}")

    # Optional transparency
    if raw_prediction < 0:
        st.info(
            "ℹ️ Linear Regression can produce negative values mathematically. "
            "The output has been adjusted to ensure a realistic price."
        )