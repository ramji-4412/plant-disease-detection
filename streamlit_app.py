import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# Class names for predictions
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load ONNX model (cached)
@st.cache_resource
def load_model():
    return ort.InferenceSession('model.onnx')

session = load_model()

def predict_disease(img):
    # Preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Run inference
    inputs = {session.get_inputs()[0].name: img_array}
    outputs = session.run(None, inputs)
    prediction = outputs[0]

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return predicted_class, confidence

# Streamlit app
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Home", "About", "Detect Disease"])

if page == "Home":
    st.title("🌱 Plant Disease Detection")
    st.write("""
    Welcome to the Plant Disease Detection app! This tool uses AI to identify diseases in plant leaves.
    
    **How to use:**
    1. Go to the "Detect Disease" page
    2. Upload a clear photo of a plant leaf
    3. Get instant disease diagnosis
    
    **Supported plants:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
    """)

elif page == "About":
    st.title("About")
    st.write("""
    This app uses a deep learning model trained on thousands of plant leaf images to detect various diseases.
    
    **Model Details:**
    - Framework: TensorFlow/Keras
    - Architecture: Convolutional Neural Network
    - Accuracy: High accuracy on trained classes
    
    **Disclaimer:** This is a diagnostic tool and should not replace professional agricultural advice.
    """)

elif page == "Detect Disease":
    st.title("🔍 Detect Plant Disease")
    st.write("Upload an image of a plant leaf to detect diseases.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Analyze Disease"):
            with st.spinner("Analyzing..."):
                predicted_class, confidence = predict_disease(image)

            st.success("Analysis Complete!")

            # Display results
            st.subheader("Prediction Results:")
            st.write(f"**Predicted Disease:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            # Health status
            if "healthy" in predicted_class.lower():
                st.success("✅ The plant appears to be healthy!")
            else:
                st.error("⚠️ Disease detected. Consider consulting an agricultural expert.")

            # Additional info
            st.info("Note: This is an AI prediction. For accurate diagnosis, consult a professional.")