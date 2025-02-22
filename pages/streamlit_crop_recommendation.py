import streamlit as st
import numpy as np
import pickle  

# Apply custom CSS for layout styling
st.markdown(
    """
    <style>
    /* Reduce input field width and align left */
    .stTextInput>div>div>input {
        width: 250px !important;  /* Adjust as needed */
    }
    /* Style the prediction box */
    .prediction-box {
        border: 2px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #333;
    }
    /* Align input fields to the left */
    .stApp {
        display: flex;
        flex-direction: row;
    }
    .left-column {
        flex: 1;
        padding-right: 50px;
    }
    .right-column {
        flex: 2;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🌱 Crop Recommendation System")

# Create layout
col1, col2 = st.columns([1, 2])

with col1:
    st.write("### Enter Soil & Climate Parameters:")
    N_input = st.number_input("Nitrogen (N)",value =99 ,placeholder="0-100")
    P_input = st.number_input("Phosphorus (P)", value =50 ,placeholder="0-100")
    K_input = st.number_input("Potassium (K)",  value =15 ,placeholder="0-100")
    Temperature_input = st.number_input("Temperature (°C)",value =18.4 ,placeholder="-10 to 50")
    Humidity_input = st.number_input("Humidity (%)", value =71 , placeholder="0-100")
    pH_input = st.number_input("pH Level",value =5.5 ,placeholder="0-14")
    Rainfall_input = st.number_input("Rainfall (mm)",value =88 ,  placeholder="0-500")


with col2:
    prediction_box_placeholder = st.empty()

if st.button("🌾 Predict Crop"):
    try:
        # Convert inputs to float
        N = float(N_input)
        P = float(P_input)
        K = float(K_input)
        Temperature = float(Temperature_input)
        Humidity = float(Humidity_input)
        pH = float(pH_input)
        Rainfall = float(Rainfall_input)

        # Load the trained model
        with open("qda_model.pkl", "rb") as model_file:
            qda = pickle.load(model_file)

        # Load the label encoder
        with open("label_encoder.pkl", "rb") as encoder_file:
            label_encoder = pickle.load(encoder_file)

        # Make prediction
        new_sample = np.array([[N, P, K, Temperature, Humidity, pH, Rainfall]])
        predicted_crop_encoded = qda.predict(new_sample)
        predicted_crop = label_encoder.inverse_transform(predicted_crop_encoded)

        # Display in styled box
        prediction_box_placeholder.markdown(
            f"""
            <div class="prediction-box">
                🌾 <b>Recommended Crop:</b> {predicted_crop[0]}
            </div>
            """,
            unsafe_allow_html=True
        )

    except ValueError:
        st.error("🚨 Invalid input! Please enter numeric values.")





# import streamlit as st
# import numpy as np
# import pickle  # Load the trained model and encoder

# # Load the trained QDA model
# with open("qda_model.pkl", "rb") as model_file:
#     qda = pickle.load(model_file)

# # Load the label encoder
# with open("label_encoder.pkl", "rb") as encoder_file:
#     label_encoder = pickle.load(encoder_file)

# # Streamlit App UI
# st.title("Crop Recommendation System")
# st.write("Enter the values below to predict the suitable crop:")

# # Create input fields for user input
# N = st.number_input("Nitrogen (N)",value =None ,placeholder="0-100")
# P = st.number_input("Phosphorus (P)", value =None ,placeholder="0-100")
# K = st.number_input("Potassium (K)",  value =None ,placeholder="0-100")
# Temperature = st.number_input("Temperature (°C)",value =None ,placeholder="-10 to 50")
# Humidity = st.number_input("Humidity (%)", value =None , placeholder="0-100")
# pH = st.number_input("pH Level",value =None ,placeholder="0-14")
# Rainfall = st.number_input("Rainfall (mm)",value =None ,  placeholder="0-500")   

# # Prediction button
# if st.button("Predict Crop"):
#     new_sample = np.array([[N, P, K, Temperature, Humidity, pH, Rainfall]])
#     predicted_crop_encoded = qda.predict(new_sample)
#     predicted_crop = label_encoder.inverse_transform(predicted_crop_encoded)
#     st.success(f"Predicted Crop: {predicted_crop[0]}")



