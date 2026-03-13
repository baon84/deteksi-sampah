import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Green Li AI", layout="centered")

st.title("🌱 Green Li AI")
st.write("Deteksi sampah pintar untuk bumi yang lebih hijau.")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

model = load_model()
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Input Kamera di HP/Laptop
img_file = st.camera_input("Ambil Foto Sampah")

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
    st.write(f"Keyakinan: {score*100:.1f}%")