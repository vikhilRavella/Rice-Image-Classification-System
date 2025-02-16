import os
import requests
import tensorflow as tf

# Replace with your actual GitHub Release download link
MODEL_URL = "https://github.com/vikhilRavella/Rice-Image-Classification-System/releases/download/v1.0/rice_classification_model.h5"
MODEL_PATH = "rice_classification_model.h5"

import os
import streamlit as st

MODEL_PATH = "rice_classification_model.h5"

MODEL_PATH = "rice_classification_model.h5"

st.title("Rice Classification Model Checker")

if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH)
    st.success(f"✅ Model file exists.\n📁 File size: {file_size / (1024*1024):.2f} MB")
else:
    st.error("❌ Model file is missing! Please upload or check the file path.")
