import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦")

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    # Ensure the model file name matches exactly what you saved in your training script
    model_path = 'traffic_classifier.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model file '{model_path}' not found. Please ensure it is in the same directory.")
        return None

model = load_my_model()

# --- LABEL MAPPING ---
classes = { 
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 
    41:'End of no passing', 42:'End no passing veh > 3.5 tons' 
}

# --- USER INTERFACE ---
st.title("🚦 Traffic Sign Recognition")
st.markdown("""
Upload an image of a German traffic sign, and the neural network will predict its class.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # 1. Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # 2. Preprocessing
    # We must resize to (30, 30) and normalize by 255.0 to match training data
    img = image.convert('RGB') # Ensure 3 channels
    img = img.resize((30, 30))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0) # Add batch dimension: (1, 30, 30, 3)

    # 3. Prediction
    with st.spinner('Classifying...'):
        preds = model.predict(img)
        class_idx = np.argmax(preds)
        probability = np.max(preds)

    # 4. Results
    st.divider()
    st.subheader(f"Prediction: **{classes[class_idx]}**")
    st.progress(float(probability))
    st.write(f"Confidence: {probability:.2%}")

    # Optional: Top 3 Probabilities
    with st.expander("View Top 3 Probabilities"):
        top_3_indices = np.argsort(preds[0])[-3:][::-1]
        for i in top_3_indices:
            st.write(f"{classes[i]}: {preds[0][i]:.2%}")

elif model is None:
    st.warning("Please upload the 'traffic_classifier.h5' file to the project folder to begin.")
