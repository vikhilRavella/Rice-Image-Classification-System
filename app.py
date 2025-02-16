import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image    

# Load Pretrained Model

import gdown
import tensorflow as tf
import os


url = "https://drive.google.com/file/d/18WPJaUejS7UFoBcdtfxNZf4fKVosbQZn/view?usp=drive_link"
gdown.download(url, model_path, quiet=False)

model = tf.keras.models.load_model(model_path)

# Define Rice Class Labels
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

def is_blurry(image):
    """Detect if an image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100  

def detect_rice_grains(image):
    """Detect rice grains in an image using contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rice_grains = [c for c in contours if 500 < cv2.contourArea(c) < 50000]
    return rice_grains

def extract_and_predict_rice_grains(image):
    """Extract and classify rice grains from an image."""
    if is_blurry(image):
        return "Blurry Image", None
    
    rice_grains = detect_rice_grains(image)
    if len(rice_grains) < 2:
        return None, None

    rice_images = []

    for contour in rice_grains:
        x, y, w, h = cv2.boundingRect(contour)
        grain_cropped = image[y:y+h, x:x+w]
        grain_resized = cv2.resize(grain_cropped, (150, 150))
        rice_images.append(grain_resized)

    if not rice_images:
        return None, None

    rice_images = np.array(rice_images) / 255.0
    rice_images = np.expand_dims(rice_images, axis=-1) * np.ones(3, dtype=int)  # Convert grayscale to 3-channel
    
    predictions = model.predict(rice_images)
    predicted_classes = [np.argmax(pred) for pred in predictions]
    most_common_class = Counter(predicted_classes).most_common(1)[0][0]
    best_class_name = class_names[most_common_class]

    rice_grain_uint8 = (rice_images[0] * 255).astype(np.uint8)
    rice_grain_uint8 = cv2.cvtColor(rice_grain_uint8, cv2.COLOR_RGB2BGR)

    return best_class_name, rice_grain_uint8

# Streamlit App
st.title("🍚 Rice Grain Classifier")
uploaded_file = st.file_uploader("Upload a rice image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    best_rice_class, rice_grain_image = extract_and_predict_rice_grains(image)

    if best_rice_class and rice_grain_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
        with col2:
            st.image(rice_grain_image, caption=f"🔍 Predicted: {best_rice_class}", use_column_width=True)
        st.success(f"✅ Best Predicted Rice Class: {best_rice_class}")
    else:
        st.error("❌ No valid rice grains detected. Please upload a clear rice image.")
