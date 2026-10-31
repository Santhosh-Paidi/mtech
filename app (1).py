import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

# Load the pre-trained logistic regression model
@st.cache_resource
def load_model():
    return joblib.load("logistic.sav")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading the model 'logistic.sav': {e}")
    model_loaded = False

# App title and description
st.title("🚚 Delivery Delay Predictor")
st.write("Enter the delivery details below to predict if the shipment will be delayed.")

if model_loaded:
    # Feature input forms based on dataset columns
    col1, col2 = st.columns(2)

    with col1:
        delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, value=20.0, step=0.1)
        traffic_congestion = st.slider("Traffic Congestion Level (1-5)", min_value=1, max_value=5, value=3)
        weather_condition = st.slider("Weather Condition (1-5)", min_value=1, max_value=5, value=2)
        delivery_slot = st.slider("Delivery Slot Hour/Session (1-4)", min_value=1, max_value=4, value=2)
        driver_experience = st.number_input("Driver Experience (Years)", min_value=0, value=5, step=1)
        num_stops = st.number_input("Number of Stops", min_value=0, value=2, step=1)

    with col2:
        vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, value=3, step=1)
        road_condition_score = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=4)
        package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, value=15.0, step=0.1)
        fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=0.0, value=12.0, step=0.1)
        warehouse_processing_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, value=45, step=1)

    # Predict button
    if st.button("Predict Delay"):
        features = np.array([[
            delivery_distance, 
            traffic_congestion, 
            weather_condition, 
            delivery_slot, 
            driver_experience, 
            num_stops, 
            vehicle_age, 
            road_condition_score, 
            package_weight, 
            fuel_efficiency, 
            warehouse_processing_time
        ]])
        
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0][1]
        
        st.write("---")
        if prediction == 1:
            st.error(f"🚨 **High risk of delay!** (Probability: {prediction_proba:.2%})")
        else:
            st.success(f"✅ **On-time delivery likely.** (Probability of delay: {prediction_proba:.2%})")
