import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import tensorflow as tf
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
CITRA = os.path.join(BASE, 'contoh_uji.jpg')
PEMANASAN = 3
PENGULANGAN = 30

# Bangun arsitektur dan muat bobot
def build_and_initialize(m_type):
    if m_type == 'mobile':
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    else:
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    m = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(15, activation='softmax')
    ])
    m.build((None, 224, 224, 3))
    return m


print("Memuat model...")
model_mobilenet = build_and_initialize('mobile')
model_efficientnet = build_and_initialize('eff')
model_mobilenet.load_weights(os.path.join(BASE, 'mobile.weights.h5'))
model_efficientnet.load_weights(os.path.join(BASE, 'eff.weights.h5'))
print("Model berhasil dimuat.\n")


# Siapkan citra uji
img = Image.open(CITRA).convert('RGB')
arr = tf.keras.preprocessing.image.img_to_array(
    img.resize((224, 224), Image.NEAREST))

x_mob = np.expand_dims(arr.copy() / 255.0, 0)   # MobileNetV2 butuh rescale
x_eff = np.expand_dims(arr.copy(), 0)           # EfficientNet-B0 tanpa rescale


# Ukur waktu inferensi
print(f"Perangkat  : CPU (GPU tidak digunakan)")
print(f"Pemanasan  : {PEMANASAN} iterasi")
print(f"Pengulangan: {PENGULANGAN} iterasi\n")
print(f"{'Arsitektur':<18}{'Rata-rata':>12}{'Minimum':>12}"
      f"{'Maksimum':>12}{'Simpangan':>12}")
print("-" * 66)

hasil = {}
for nama, model, x in [('MobileNetV2', model_mobilenet, x_mob),
                       ('EfficientNet-B0', model_efficientnet, x_eff)]:

    for _ in range(PEMANASAN):
        model.predict(x, verbose=0)

    waktu = []
    for _ in range(PENGULANGAN):
        t = time.perf_counter()
        model.predict(x, verbose=0)
        waktu.append((time.perf_counter() - t) * 1000)

    hasil[nama] = np.mean(waktu)
    print(f"{nama:<18}{np.mean(waktu):>9.1f} ms{np.min(waktu):>9.1f} ms"
          f"{np.max(waktu):>9.1f} ms{np.std(waktu):>9.1f} ms")

print("-" * 66)
rasio = hasil['EfficientNet-B0'] / hasil['MobileNetV2']
selisih = hasil['EfficientNet-B0'] - hasil['MobileNetV2']
print(f"\nEfficientNet-B0 membutuhkan waktu {rasio:.2f} kali lipat "
      f"dibandingkan MobileNetV2")
print(f"Selisih absolut: {selisih:.1f} ms per citra")
print(f"Total waktu inferensi kedua model: "
      f"{hasil['MobileNetV2'] + hasil['EfficientNet-B0']:.1f} ms per citra")