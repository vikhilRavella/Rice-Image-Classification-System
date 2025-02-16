import os
import requests
import tensorflow as tf

# Replace with your actual GitHub Release download link
MODEL_URL = "https://github.com/vikhilRavella/Rice-Image-Classification-System/releases/download/v1.0/rice_classification_model.h5"
MODEL_PATH = "rice_classification_model.h5"

import os
import streamlit as st
import tensorflow as tf

# Check if the file exists
if os.path.exists(MODEL_PATH):
    st.write("Model file exists.")
    st.write("File size:", os.path.getsize(MODEL_PATH), "bytes")
    
    # Try loading the model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.write("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

else:
    st.error("Model file is missing! Upload the model file and try again.")
