import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fused_cnn_model.keras')

# Initialize label encoder
@st.cache_resource
def get_label_encoder():
    le = LabelEncoder()
    le.fit(['Healthy', 'Tumor'])
    return le

def predict_image(model, image, target_size=(224, 224)):
    img = image.resize(target_size)
    arr = img_to_array(img) / 255.0
    fused = np.expand_dims((arr + arr) / 2.0, axis=0).astype(np.float32)
    prob = model.predict(fused, verbose=0)[0][0]
    return prob

st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain scan image (MRI or CT) to detect if it shows signs of a tumor.")

model = load_model()
label_encoder = get_label_encoder()

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            prob = predict_image(model, image)
            label = label_encoder.inverse_transform([int(prob > 0.5)])[0]
            
            st.subheader("Results:")
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {prob:.1%}" if label == "Tumor" else f"**Confidence:** {(1-prob):.1%}")
            
            if label == "Tumor":
                st.error("⚠️ Tumor detected. Please consult a medical professional.")
            else:
                st.success("✅ No tumor detected.")