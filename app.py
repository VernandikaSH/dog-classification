import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Ras Anjing - Top 3", layout="wide")

st.title("Perbandingan Klasifikasi Ras Anjing")
st.write("Aplikasi ini membandingkan arsitektur MobileNetV2 dan EfficientNet-B0.")

@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)

    def build_and_initialize(m_type):
        if m_type == 'mobile':
            base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
        else:
            base = tf.keras.applications.EfficientNetB0(input_shape=(224,224,3), include_top=False, weights='imagenet')
        
        m = tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(15, activation='softmax')
        ])
        m.build((None, 224, 224, 3))
        return m

    model_mobile = build_and_initialize('mobile')
    model_eff = build_and_initialize('eff')

    try:
        model_mobile.load_weights(os.path.join(base_path, 'mobile.weights.h5'))
        model_eff.load_weights(os.path.join(base_path, 'eff.weights.h5'))
        return model_mobile, model_eff
    except Exception as e:
        st.error(f"Gagal memuat file bobot: {e}")
        return None, None

# Inisialisasi model
model_mobilenet = None
model_efficientnet = None

with st.spinner("Memuat model, mohon tunggu..."):
    try:
        model_mobilenet, model_efficientnet = load_models()
    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")

# Daftar kelas (urutan sesuai folder Kaggle)
classes = ['beagle', 'boxer', 'chihuahua', 'cocker_spaniel', 'doberman', 
           'french_bulldog', 'german_shepherd', 'golden_retriever', 
           'labrador_retriever', 'malamute', 'pug', 'rottweiler', 
           'samoyed', 'siberian_husky', 'tzu']

# Mapping nama tampilan untuk pengguna
CLASS_DISPLAY_NAMES = {
    'beagle': 'Beagle',
    'boxer': 'Boxer',
    'chihuahua': 'Chihuahua',
    'cocker_spaniel': 'Cocker Spaniel',
    'doberman': 'Doberman',
    'french_bulldog': 'French Bulldog',
    'german_shepherd': 'German Shepherd',
    'golden_retriever': 'Golden Retriever',
    'labrador_retriever': 'Labrador Retriever',
    'malamute': 'Malamute',
    'pug': 'Pug',
    'rottweiler': 'Rottweiler',
    'samoyed': 'Samoyed',
    'siberian_husky': 'Siberian Husky',
    'tzu': 'Shih Tzu'
}

CONFIDENCE_THRESHOLD = 0.50  

# --- PENGUJIAN GAMBAR ---
st.header("Pengujian Gambar Baru")
uploaded_file = st.file_uploader("Pilih gambar anjing...", type=["jpg", "jpeg", "png"])

MAX_MB = 10          # batas ukuran berkas yang diterima
MIN_PIXEL = 32       # dimensi minimum agar citra layak diproses
 
if uploaded_file is not None:
 
    # --- Validasi 1: ukuran berkas ---
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_MB:
        st.error(f"❌ Ukuran berkas {size_mb:.1f} MB melebihi batas {MAX_MB} MB.")
        st.stop()
 
    if uploaded_file.size == 0:
        st.error("❌ Berkas kosong. Silakan unggah berkas citra yang valid.")
        st.stop()
 
    # --- Validasi 2: integritas berkas citra ---
    try:
        img = Image.open(uploaded_file)
        img.verify()                 # cek struktur berkas tanpa memuat piksel
        uploaded_file.seek(0)        # verify() menutup berkas, buka ulang
        img = Image.open(uploaded_file)
        img = img.convert('RGB')     # tangani RGBA, grayscale, palet
    except Exception:
        st.error("❌ Berkas tidak dapat dibaca sebagai citra.")
        st.info(
            "Pastikan berkas yang diunggah benar-benar berupa gambar "
            "berformat JPG, JPEG, atau PNG dan tidak dalam kondisi rusak."
        )
        st.stop()
 
    # --- Validasi 3: dimensi minimum ---
    if img.width < MIN_PIXEL or img.height < MIN_PIXEL:
        st.error(
            f"❌ Resolusi citra terlalu kecil ({img.width}×{img.height} piksel). "
            f"Minimum {MIN_PIXEL}×{MIN_PIXEL} piksel."
        )
        st.stop()
 
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
 
    main_col1, main_col2 = st.columns([1, 2])
 
    with main_col1:
        st.image(img, caption="Gambar Input", use_container_width=True)
 
    img_resized = img.resize((224, 224), Image.NEAREST)   # selaras dgn pelatihan
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)

    if model_mobilenet is not None and model_efficientnet is not None:
        with main_col1:
            execute_button = st.button("Jalankan Klasifikasi", use_container_width=True)
            
        if execute_button:
            with st.spinner("Menganalisis gambar..."):
                try:
                    # Prediksi kedua model
                    img_mobile = np.expand_dims(img_array.copy() / 255.0, axis=0)
                    res_mobile = model_mobilenet.predict(img_mobile)
                    
                    img_eff = np.expand_dims(img_array.copy(), axis=0)
                    res_eff = model_efficientnet.predict(img_eff)

                    # Cek apakah gambar kemungkinan anjing (rata-rata keyakinan)
                    avg_confidence = (np.max(res_mobile) + np.max(res_eff)) / 2

                    if avg_confidence < CONFIDENCE_THRESHOLD:
                        with main_col2:
                            st.error("⚠️ Gambar tidak terdeteksi sebagai anjing.")
                            st.warning(
                                f"Tingkat keyakinan model terlalu rendah "
                                f"(rata-rata: {avg_confidence*100:.1f}% dari "
                                f"minimum {CONFIDENCE_THRESHOLD*100:.0f}% yang dibutuhkan). "
                                f"Pastikan gambar menampilkan anjing dengan jelas."
                            )
                            st.info(
                                "💡 Tips: Gunakan foto anjing yang jelas, "
                                "pencahayaan cukup, dan anjing menjadi objek utama dalam foto."
                            )
                    else:
                        # Gambar terdeteksi sebagai anjing — tampilkan hasil
                        idx_mobile = np.argsort(res_mobile[0])[-3:][::-1]
                        idx_eff = np.argsort(res_eff[0])[-3:][::-1]

                        with main_col2:
                            st.subheader("Hasil Analisis Model")
                            r_col1, r_col2 = st.columns(2)
                            
                            with r_col1:
                                st.markdown("### **MobileNetV2**")
                                for i in idx_mobile:
                                    pct = res_mobile[0][i] * 100
                                    st.write(f"**{CLASS_DISPLAY_NAMES[classes[i]]}**: {pct:.2f}%")
                                    st.progress(int(pct))

                            with r_col2:
                                st.markdown("### **EfficientNet-B0**")
                                for i in idx_eff:
                                    pct = res_eff[0][i] * 100
                                    st.write(f"**{CLASS_DISPLAY_NAMES[classes[i]]}**: {pct:.2f}%")
                                    st.progress(int(pct))
                                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")