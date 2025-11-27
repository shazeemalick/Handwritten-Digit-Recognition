import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Load trained model
model = load_model("mnist_ann_model.h5")

# Nebula theme styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 30px;
        border-radius: 15px;
        color: white;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        border: 1px solid #8a2be2;
        box-shadow: 0 4px 15px rgba(138, 43, 226, 0.3);
    }
    .upload-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 1px solid #4a00e0;
        backdrop-filter: blur(10px);
    }
    .title-text {
        background: linear-gradient(45deg, #ff6b6b, #f8e71c, #7ed321, #4a90e2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stImage {
        border-radius: 10px;
        border: 2px solid #8a2be2;
    }
</style>
""", unsafe_allow_html=True)

# # Main container with nebula background
# st.markdown('<div class="main">', unsafe_allow_html=True)

# Cosmic title
st.markdown('<div class="title-text">🌌 Digit Recognition App</div>', unsafe_allow_html=True)

# Space divider
# st.markdown("<br>", unsafe_allow_html=True)

# Upload section with glass effect
# st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload a digit image (28x28)", type=["png","jpg","jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# Your original code - completely unchanged
if uploaded_file is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28
    img = cv2.resize(img, (28,28))

    # Show uploaded image
    st.image(img, caption="✨ Uploaded Image", width=150)

    # Preprocess
    img = img.reshape(1, 784) / 255.0

    # Predict
    prediction = model.predict(img)
    result = np.argmax(prediction)

    # Nebula-themed prediction box
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown("### 🌟 Prediction Result")
    st.markdown(f"# 🔢 **{result}**")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)