import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pyfeats import zernikes_moments
from joblib import load

# Load model and label encoder
model = load("zernike_alphabet_svm_model.joblib")
label_encoder = load("alphabet_label_encoder.joblib")

st.set_page_config(page_title="Zernike Alphabet Classifier", layout="centered")
st.title("🔤 Alphabet Recognition using Zernike Moments")
st.markdown("Upload a clean **capital letter** image (128x128 recommended)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((128, 128))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image_np = np.array(image)
    _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract features
    features, _ = zernikes_moments(thresh, radius=64)

    # Predict
    prediction = model.predict([features])[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    st.success(f"🧠 Predicted Letter: **{predicted_label}**")
