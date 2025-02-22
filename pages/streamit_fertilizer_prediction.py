import streamlit as st
import pickle
import numpy as np

# Load the trained model and preprocessing objects
with open("fertilizer_model.pkl", "rb") as f:
    model_data = pickle.load(f)

voting_clf = model_data["model"]
soil_encoder = model_data["soil_encoder"]
crop_encoder = model_data["crop_encoder"]
scaler = model_data["scaler"]
fertilizer_encoder = model_data["fertilizer_encoder"]

# Streamlit App
st.title("🌾 Smart Farming: Fertilizer Recommendation System")

# User Inputs
soil_type = st.selectbox("Select Soil Type", soil_encoder.classes_)
crop_type = st.selectbox("Select Crop Type", crop_encoder.classes_)
temperature = st.number_input("Enter Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
moisture = st.number_input("Enter Moisture Level (%)", min_value=0.0, max_value=100.0, value=40.0)
nitrogen = st.number_input("Enter Nitrogen Level", min_value=0, max_value=200, value=50)
phosphorus = st.number_input("Enter Phosphorus Level", min_value=0, max_value=200, value=40)
potassium = st.number_input("Enter Potassium Level", min_value=0, max_value=200, value=30)

# Predict Fertilizer
if st.button("Predict Fertilizer"):
    # Encode categorical inputs
    soil_encoded = soil_encoder.transform([soil_type])[0]
    crop_encoded = crop_encoder.transform([crop_type])[0]

    # Prepare input data
    input_data = np.array([[soil_encoded, crop_encoded, temperature, humidity, moisture, nitrogen, phosphorus, potassium]])
    input_scaled = scaler.transform(input_data)

    # Get Prediction
    fertilizer_pred = voting_clf.predict(input_scaled)[0]
    fertilizer_name = fertilizer_encoder.inverse_transform([fertilizer_pred])[0]

    st.success(f"🚜 Recommended Fertilizer: **{fertilizer_name}**")
