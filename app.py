import streamlit as st
import cv2
import numpy as np
from PIL import Image
from aquavision.predict import AquaPredictor

st.title("AquaVision BD — Diagnostic Dashboard")

species = st.sidebar.selectbox("Select Species", ["fish", "shrimp"])
predictor = AquaPredictor(species=species, fold=0)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

st.sidebar.subheader("Water Quality Parameters")
temp = st.sidebar.slider("Temperature (°C)", 15.0, 35.0, 25.0)
ph = st.sidebar.slider("pH Level", 5.0, 10.0, 7.5)
do = st.sidebar.slider("Dissolved Oxygen (mg/L)", 1.0, 10.0, 6.0)
ammonia = st.sidebar.slider("Ammonia (mg/L)", 0.0, 3.0, 0.01)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB"))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Diagnosis"):
        wq_dict = {"temperature": temp, "ph": ph, "dissolved_oxygen": do, "ammonia": ammonia}
        result = predictor.predict(img_array, water_quality_dict=wq_dict)
        
        st.success(f"**Prediction:** {result['prediction']}")
        st.info(f"**Confidence:** {result['confidence']*100:.2f}%")
