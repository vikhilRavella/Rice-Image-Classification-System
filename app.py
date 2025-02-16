import os
import requests
import tensorflow as tf

# Replace with your actual GitHub Release download link
MODEL_URL = "https://github.com/vikhilRavella/Rice-Image-Classification-System/releases/download/v1.0/rice_classification_model.h5"
MODEL_PATH = "rice_classification_model.h5"

# Download the model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
