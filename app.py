import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
# Kode tambahan untuk menyisipkan Manifest PWA
st.markdown(
    f"""
    <link rel="manifest" href="manifest.json">
    <meta name="theme-color" content="#2E7D32">
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="Green Bin AI", layout="centered")

st.title("🌱 Green Bin AI")
st.write("AI-Based Waste Detection System.")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

model = load_model()
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Input Kamera di HP/Laptop
img_file = st.camera_input("Take a photo of the trash")

if img_file:
    image = Image.open(img_file).convert("RGB")
    
    # Preprocessing agar sesuai input AI (224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array

    # Prediksi
    prediction = model.predict(data)
    index = np.argmax(prediction)
    label = class_names[index]
    score = prediction[0][index]

    # Tampilan Hasil
    st.success(f"Terdeteksi: **{label[2:]}**")

    st.write(f"Percent : {score*100:.1f}%")






