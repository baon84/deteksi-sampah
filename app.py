import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. HARUS PALING ATAS
st.set_page_config(page_title="Green Bin AI", page_icon="🌱", layout="centered")

# 2. Menyisipkan Manifest PWA & Meta Theme
st.markdown('''
    <link rel="manifest" href="./manifest.json?v=10">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-title" content="Green Bin AI">
''', unsafe_allow_html=True)

st.title("🌱 Green Bin AI")
st.write("AI-Based Waste Detection System.")

# Load Model
@st.cache_resource
def load_model():
    # Pastikan file keras_model.h5 dan labels.txt sudah ada di GitHub
    return tf.keras.models.load_model("keras_model.h5", compile=False)

try:
    model = load_model()
    class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
except Exception as e:
    st.error("Model belum terdeteksi. Pastikan file .h5 sudah diunggah ke GitHub.")

# Input Kamera
img_file = st.camera_input("Take a photo of the trash")

if img_file:
    image = Image.open(img_file).convert("RGB")
    
    # Preprocessing (224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array

    # Prediksi
    with st.spinner('Sedang menganalisis...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        label = class_names[index]
        score = prediction[0][index]

    # Tampilan Hasil
    st.divider()
    # Mengambil teks label setelah angka (misal '0 Organik' jadi 'Organik')
    clean_label = label.split(' ', 1)[1] if ' ' in label else label
    
    st.success(f"Terdeteksi: **{clean_label}**")
    st.progress(float(score))
    st.write(f"Tingkat Keyakinan: {score*100:.1f}%")

    if "B3" in label.upper():
        st.warning("⚠️ Ini sampah B3! Tangani dengan hati-hati.")



