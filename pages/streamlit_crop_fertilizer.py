import streamlit as st
import pickle
import pandas as pd

# Load the trained models
@st.cache_data
def load_models():
    with open("qda_model.pkl", "rb") as model_file:
        qda_model = pickle.load(model_file)
    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    with open("fertilizer_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    return qda_model, label_encoder, model_data

qda_model, label_encoder, model_data = load_models()

# Extract components from fertilizer model
voting_clf = model_data["model"]
soil_encoder = model_data["soil_encoder"]
crop_encoder = model_data["crop_encoder"]
scaler = model_data["scaler"]
fertilizer_encoder = model_data["fertilizer_encoder"]

# Mapping crop types to general categories
crop_fertilizer_mapping = {
    "Paddy": ["rice"],
    "Maize": ["maize"],
    "Pulses": ["chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"],
    "Oil seeds": ["pomegranate", "banana", "mango", "grapes", "coconut"],
    "Millets": ["millets"],
    "Ground Nuts": [],
    "Sugarcane": ["sugarcane"],
    "Cotton": ["cotton"],
    "Tobacco": ["tobacco"],
    "Barley": ["barley"],
    "Wheat": ["wheat"],
    "Fruits": ["watermelon", "muskmelon", "apple", "orange", "papaya"]
}

# Function to recommend crop
def recommend_crop(features: pd.DataFrame):
    predicted_label = qda_model.predict(features)
    crop_name = label_encoder.inverse_transform(predicted_label)
    return crop_name[0] 

# Function to map crops to categories
def get_modified_crop_type(crop_type: str):
    for main_crop, crops in crop_fertilizer_mapping.items():
        if crop_type in crops:
            return main_crop
    return crop_type

# Function to recommend fertilizer
def recommend_fertilizer(crop_type: str, features: pd.DataFrame):
    modified_crop_type = get_modified_crop_type(crop_type)
    
    encoded_crop = crop_encoder.transform([modified_crop_type])
    features["Crop Type"] = encoded_crop[0]
    
    encoded_soil = soil_encoder.transform(features["Soil Type"].values.reshape(-1, 1))
    features["Soil Type"] = encoded_soil[0]
    
    scaled_features = scaler.transform(features)
    
    predicted_fertilizer = voting_clf.predict(scaled_features)
    fertilizer_name = fertilizer_encoder.inverse_transform(predicted_fertilizer)
    
    return fertilizer_name[0]

# Streamlit UI
st.title("🌱 Smart Farming: Crop & Fertilizer Prediction 🚜")
st.markdown("### Enter Soil and Environmental Parameters to get Recommendations")

# User input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=90)
P = st.number_input("Phosphorous (P)", min_value=0, max_value=150, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=150, value=43)
temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, value=20)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=82)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=500, value=200)

# Soil type selection
soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clayey", "Black", "Red"])

# Button to predict
if st.button("🌿 Get Recommendations"):
    # Prepare input data
    crop_features = pd.DataFrame({
        'N': [N], 'P': [P], 'K': [K], 'temperature': [temperature],
        'humidity': [humidity], 'ph': [ph], 'rainfall': [rainfall]
    })
    
    # Get recommended crop
    recommended_crop = recommend_crop(crop_features)
    st.success(f"🌾 Recommended Crop: **{recommended_crop}**")
    
    # Prepare input for fertilizer model
    fertilizer_features = pd.DataFrame({
        'Temperature': [temperature], 'Humidity': [humidity], 'Moisture': [30],  
        'Soil Type': [soil_type], 'Crop Type': [recommended_crop],  
        'Nitrogen': [N], 'Phosphorous': [P], 'Potassium': [K]
    })

    # Get recommended fertilizer
    recommended_fertilizer = recommend_fertilizer(recommended_crop, fertilizer_features)
    st.info(f"🌱 Recommended Fertilizer: **{recommended_fertilizer}**")

