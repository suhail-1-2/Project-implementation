import pickle
import pandas as pd

# Function for crop recommendation
def recommend_crop(features: pd.DataFrame):
    # Load the crop recommendation model and label encoder
    with open("qda_model.pkl", "rb") as model_file:
        qda_model = pickle.load(model_file)

    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    
    # Predict the crop type (label) using the QDA model
    predicted_label = qda_model.predict(features)
    
    # Decode the predicted label to the actual crop name
    crop_name = label_encoder.inverse_transform(predicted_label)
    
    return crop_name[0]  # Return the recommended crop name

# Example usage
# Example features for crop recommendation
features = pd.DataFrame({
    'N': [90],
    'P': [40],
    'K': [70],
    'temperature': [10],
    'humidity': [90],
    'ph': [9.5],
    'rainfall': [100]
})

# recommended_crop = recommend_crop(features)
# print(f"Recommended Crop: {recommended_crop}")




# Function to load the fertilizer recommendation model and predict the fertilizer
def recommend_fertilizer(crop_type: str, features: pd.DataFrame):
    # Map the crop type based on the mapping dictionary
    modified_crop_type = get_modified_crop_type(crop_type)
    
    # Load the fertilizer recommendation model and other components
    with open("fertilizer_model.pkl", "rb") as f:
        model_data = pickle.load(f)
        
        # Extract the necessary components
        voting_clf = model_data["model"]
        soil_encoder = model_data["soil_encoder"]
        crop_encoder = model_data["crop_encoder"]
        scaler = model_data["scaler"]
        fertilizer_encoder = model_data["fertilizer_encoder"]
        
    # Encode the modified crop type (since the model uses encoded crop type)
    encoded_crop = crop_encoder.transform([modified_crop_type])
    # Add the encoded crop type to the feature set
    features["Crop Type"] = encoded_crop[0]
    
    encoded_soil = soil_encoder.transform(features["Soil Type"].values.reshape(-1, 1))
    features["Soil Type"] = encoded_soil[0]
    
    # Scale the features if needed (if scaling was used during training)
    scaled_features = scaler.transform(features)
    
    # Predict the fertilizer
    predicted_fertilizer = voting_clf.predict(scaled_features)
    
    # Decode the fertilizer recommendation
    fertilizer_name = fertilizer_encoder.inverse_transform(predicted_fertilizer)
    
    return fertilizer_name[0]  # Return the recommended fertilizer

# Function to get the modified crop type based on the mapping
def get_modified_crop_type(crop_type: str):
    # Find the corresponding modified crop type from the mapping
    # Crop Fertilizer Mapping
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


    for main_crop, crops in crop_fertilizer_mapping.items():
        if crop_type in crops:
            return main_crop
    return crop_type  # Return the original crop if no mapping is found

# Get the crop recommendation (for example, it might return "Lentil")
recommended_crop = recommend_crop(features)

# Prepare the input features for fertilizer prediction (ensure all necessary columns are present)
fertilizer_features = pd.DataFrame({
    'Temperature': [30],
    'Humidity': [60],
    'Moisture': [30],
    'Soil Type': ['Loamy'],  # Make sure to encode soil type before passing if required
    'Crop Type': [recommended_crop],  # Crop type comes from the previous model
    'Nitrogen': [120],
    'Phosphorous': [50],
    'Potassium': [80]
})

# Get the recommended fertilizer based on the mapped crop type (e.g., "Pulses" instead of "Lentil")
recommended_fertilizer = recommend_fertilizer(recommended_crop, fertilizer_features)
print(f"Recommended Fertilizer: {recommended_fertilizer}")