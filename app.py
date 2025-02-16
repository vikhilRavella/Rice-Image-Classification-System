import os
import requests
import tensorflow as tf

# Replace with your actual GitHub Release download link
MODEL_URL = "https://github.com/vikhilRavella/Rice-Image-Classification-System/releases/download/v1.0/rice_classification_model.h5"
MODEL_PATH = "rice_classification_model.h5"

import os

MODEL_PATH = "rice_classification_model.h5"

if os.path.exists(MODEL_PATH):
    print("Model file exists.")
    print("File size:", os.path.getsize(MODEL_PATH), "bytes")
else:
    print("Model file is missing!")
