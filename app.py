import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import load as joblib_load

# -----------------------
# CONFIG
# -----------------------

IMAGE_SIZE = (224, 224)   # change this if your CNN used different dimensions

CLASS_NAMES = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment"
]

# -----------------------
# LOAD MODELS (LOCAL)
# -----------------------

@st.cache_resource
def load_feature_extractor():
    model = load_model("feature_extractor.keras")
    return model

@st.cache_resource
def load_rf_model():
    return joblib_load("best_random_forest_model.joblib")

# -----------------------
# PREPROCESSING
# -----------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# -----------------------
# PREDICTION
# -----------------------

def predict(image):
    # load models
    feature_extractor = load_feature_extractor()
    rf_model = load_rf_model()

    # preprocess
    batch = preprocess_image(image)

    # extract features
    features = feature_extractor.predict(batch)

    # flatten for RandomForest
    features_flat = features.reshape(1, -1)

    # RF prediction
    pred_idx = int(rf_model.predict(features_flat)[0])
    probs = rf_model.predict_proba(features_flat)[0]

    return pred_idx, probs

# -----------------------
# UI
# -----------------------

st.title("🧠 Alzheimer's MRI Classifier – Local Version")
st.write("Upload an MRI image to classify it using the locally stored model.")

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Running model..."):
            pred_idx, probs = predict(img)

        st.success(f"Prediction: **{CLASS_NAMES[pred_idx]}**")

        st.write("Probabilities:")
        for i, cls in enumerate(CLASS_NAMES):
            st.write(f"{cls}: {probs[i]*100:.2f}%")
