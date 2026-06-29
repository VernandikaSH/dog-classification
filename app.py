import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Ras Anjing - Top 3", layout="wide")

st.title("Perbandingan Klasifikasi Ras Anjing")
st.write("Aplikasi ini membandingkan arsitektur MobileNetV2 dan EfficientNet-B0.")

@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)

    def build_and_initialize(m_type):
        # 1. Definisi Arsitektur
        if m_type == 'mobile':
            base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
        else:
            base = tf.keras.applications.EfficientNetB0(input_shape=(224,224,3), include_top=False, weights=None)
        
        m = tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(15, activation='softmax')
        ])

        m.build((None, 224, 224, 3)) 
        return m

    # Inisialisasi arsitektur yang sudah dibuilt
    model_mobile = build_and_initialize('mobile')
    model_eff = build_and_initialize('eff')

    # 3. Masukkan bobot dari file .weights.h5 
    try:
        model_mobile.load_weights(os.path.join(base_path, 'mobile.weights.h5'))
        model_eff.load_weights(os.path.join(base_path, 'eff.weights.h5'))
        return model_mobile, model_eff
    except Exception as e:
        # Jika nama file berbeda, sesuaikan di sini
        st.error(f"Gagal memuat file bobot: {e}")
        return None, None

# Inisialisasi variabel di luar try-except
model_mobilenet = None
model_efficientnet = None

try:
    model_mobilenet, model_efficientnet = load_models()
    st.success("Model Berhasil Dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")
    st.info("Saran: Tekan tombol 'C' pada keyboard untuk membersihkan cache Streamlit.")

# Daftar Kelas - Pastikan menggunakan underscore sesuai format folder Kaggle
classes = ['beagle', 'boxer', 'chihuahua', 'cocker_spaniel', 'doberman', 'french_bulldog', 'german_shepherd', 'golden_retriever', 'labrador_retriever', 'malamute', 'pug', 'rottweiler', 'samoyed', 'siberian_husky', 'tzu']

# --- 2. TAMPILAN GRAFIK PERFORMA ---
st.header("Performa Arsitektur")
col1, col2 = st.columns(2)

with col1:
    st.subheader("MobileNetV2")
    if os.path.exists("grafik_mobilenet.png"):
        st.image("grafik_mobilenet.png")
    st.write("**Top-1 Acc:** ~90% | **Top-3 Acc:** ~97%")

with col2:
    st.subheader("EfficientNet-B0")
    if os.path.exists("grafik_efficientnet.png"):
        st.image("grafik_efficientnet.png")
    st.write("**Top-1 Acc:** ~94% | **Top-3 Acc:** ~99%")

st.divider()

# --- 3. PENGUJIAN GAMBAR ---
st.header("Pengujian Gambar Baru")
uploaded_file = st.file_uploader("Pilih gambar anjing...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # --- PENGATURAN TATA LETAK UTAMA ---
    main_col1, main_col2 = st.columns([1, 2]) # Rasio lebar kolom 1:2
    
    with main_col1:
        st.image(img, caption="Gambar Input", use_column_width=True)
    
    # Preprocessing Dasar
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)

    if model_mobilenet is not None and model_efficientnet is not None:
        with main_col1:
            execute_button = st.button("Jalankan Klasifikasi", use_container_width=True)
            
        if execute_button:
            with st.spinner("Loading..."):
                try:
                    # 1. Prediksi MobileNetV2 (Butuh Rescale 1/255)
                    img_mobile = np.expand_dims(img_array.copy() / 255.0, axis=0)
                    res_mobile = model_mobilenet.predict(img_mobile)
                    idx_mobile = np.argsort(res_mobile[0])[-3:][::-1]
                    
                    # 2. Prediksi EfficientNet-B0 (TIDAK BOLEH Rescale)
                    img_eff = np.expand_dims(img_array.copy(), axis=0)
                    res_eff = model_efficientnet.predict(img_eff)
                    idx_eff = np.argsort(res_eff[0])[-3:][::-1]

                    # TAMPILAN HASIL 
                    with main_col2:
                        st.subheader("Hasil Analisis Model")
                        
                        # Membuat sub-kolom untuk memisahkan MobileNet vs EfficientNet
                        r_col1, r_col2 = st.columns(2)
                        
                        with r_col1:
                            st.markdown("### **MobileNetV2**")
                            for i in idx_mobile:
                                pct = res_mobile[0][i] * 100
                                st.write(f"**{classes[i]}**: {pct:.2f}%")
                                st.progress(int(pct))

                        with r_col2:
                            st.markdown("### **EfficientNet-B0**")
                            for i in idx_eff:
                                pct = res_eff[0][i] * 100
                                st.write(f"**{classes[i]}**: {pct:.2f}%")
                                st.progress(int(pct))
                                
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")