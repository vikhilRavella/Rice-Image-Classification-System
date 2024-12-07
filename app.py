import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model
model = load_model("model/rice1_classifier_model_with_rmsprop.h5")

# Class labels
class_labels = ['ardorio', 'basmati', 'ipsala', 'jasmine', 'karacadag']

# Function to predict rice type
def predict_rice(image_file):
    # Open the uploaded image
    img = Image.open(image_file)
    
    # Convert RGBA to RGB if the image has 4 channels
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize and preprocess the image
    img = img.resize((224, 224))  # Adjust to your model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values (if model expects normalized input)
    
    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] == 1:  # If the image is grayscale, convert it to RGB
        img_array = np.repeat(img_array, 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get the predicted class index and the confidence (probability)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100  # Confidence percentage
    
    return predicted_class, confidence

# Streamlit UI with custom CSS
st.markdown("""
    <style>
        .stTitle {
            font-family: 'Arial', sans-serif;
            color: #4CAF50;
            font-size: 30px;
        }
        .stText {
            font-family: 'Arial', sans-serif;
            color: #555;
            font-size: 18px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stImage>img {
            border: 10px solid #4CAF50;
            border-radius: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease-in-out;
        }
        .stImage>img:hover {
            transform: scale(1.05);
        }
        .result {
            font-family: 'Arial', sans-serif;
            font-size: 18px;
            color: #333;
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .footer {
            font-family: 'Arial', sans-serif;
            font-size: 16px;
            color: #888;
            margin-top: 50px;
            text-align: center;
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #ffffff;
            padding: 10px 0;
            border-top: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app content
st.title("Rice Type Classifier", anchor="stTitle")
st.write("Upload a rice image to classify.", anchor="stText")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

# Process uploaded file
if uploaded_file:
    # Display the uploaded image with unique style
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    predicted_class, confidence = predict_rice(uploaded_file)
    
    # Display result with confidence
    st.markdown(f"<div class='result'><strong>The predicted rice type is: </strong>{predicted_class}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result'><strong>Confidence: </strong>{confidence:.2f}%</div>", unsafe_allow_html=True)

# Footer for project leader details
st.markdown("""
    <div class="footer">
        <strong>Project Leader:</strong> Ravella Vikhil <br>
        <strong>Email:</strong> <a href="mailto:ravellavikhil2004@gmail.com">ravellavikhil2004@gmail.com</a>
    </div>
""", unsafe_allow_html=True)
