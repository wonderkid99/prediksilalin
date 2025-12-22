import streamlit as st
import numpy as np
from keras.models import load_model

st.title("🚦 Traffic Prediction AI")

model = load_model('model_lalin.h5')

speed = st.number_input("Kecepatan Saat Ini (km/h)", 0, 100, 30)
rain = st.slider("Curah Hujan (mm)", 0, 50, 0)
holiday = st.selectbox("Hari Libur?", [0, 1])

if st.button("Prediksi"):
    # ... logika preprocessing ...
    st.success(f"Prediksi Kecepatan Nanti: {hasil} km/jam")