import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

# Muat kedua model
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
        tf.keras.layers.Dense(15, activation='softmax')])
    m.build((None, 224, 224, 3))
    return m

model_mobile = build_and_initialize('mobile')
model_eff    = build_and_initialize('eff')
model_mobile.load_weights('mobile.weights.h5')
model_eff.load_weights('eff.weights.h5')

classes = ['beagle', 'boxer', 'chihuahua', 'cocker_spaniel', 'doberman',
           'french_bulldog', 'german_shepherd', 'golden_retriever',
           'labrador_retriever', 'malamute', 'pug', 'rottweiler',
           'samoyed', 'siberian_husky', 'tzu']

THRESHOLD = 0.50
ROOT = 'uji_ood'

# Fungsi inferensi
def prediksi(path):
    img = Image.open(path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    arr = tf.keras.preprocessing.image.img_to_array(
        img.resize((224, 224), Image.NEAREST))

    p_mob = model_mobile.predict(
        np.expand_dims(arr.copy() / 255.0, 0), verbose=0)[0]
    p_eff = model_eff.predict(
        np.expand_dims(arr.copy(), 0), verbose=0)[0]

    avg_conf = (p_mob.max() + p_eff.max()) / 2
    return avg_conf, classes[p_mob.argmax()], classes[p_eff.argmax()]

# Jalankan pengujian per kategori

rekap, detail = [], []

for kategori in sorted(os.listdir(ROOT)):
    folder = os.path.join(ROOT, kategori)
    if not os.path.isdir(folder):
        continue

    berkas = [f for f in sorted(os.listdir(folder))
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not berkas:
        continue

    confs, ditolak = [], 0
    for f in berkas:
        try:
            c, lm, le = prediksi(os.path.join(folder, f))
        except Exception as e:
            print(f"  ! gagal membaca {f}: {e}")
            continue
        confs.append(c)
        tolak = c < THRESHOLD
        ditolak += tolak
        detail.append({'Kategori': kategori, 'Berkas': f,
                       'Keyakinan (%)': round(c * 100, 2),
                       'Prediksi MobileNetV2': lm,
                       'Prediksi EfficientNet-B0': le,
                       'Ditolak': 'Ya' if tolak else 'Tidak'})

    n = len(confs)
    rekap.append({
        'Kategori': kategori,
        'Jumlah Citra': n,
        'Ditolak': ditolak,
        'Lolos Ambang': n - ditolak,
        'Tingkat Penolakan (%)': round(ditolak / n * 100, 1),
        'Keyakinan Rata-rata (%)': round(np.mean(confs) * 100, 1),
        'Keyakinan Tertinggi (%)': round(np.max(confs) * 100, 1),
    })

df_rekap  = pd.DataFrame(rekap)
df_detail = pd.DataFrame(detail)

print("\n===== REKAPITULASI PENGUJIAN OUT-OF-DISTRIBUTION =====")
print(df_rekap.to_string(index=False))
df_rekap.to_csv('hasil_ood_rekap.csv', index=False)
df_detail.to_csv('hasil_ood_detail.csv', index=False)

# Analisis sensitivitas ambang keyakinan
print("\n===== SENSITIVITAS AMBANG KEYAKINAN =====")
baris = []
for t in [0.40, 0.50, 0.60, 0.70, 0.80]:
    r = {'Ambang': f"{t*100:.0f}%"}
    for kategori in df_detail['Kategori'].unique():
        sub = df_detail[df_detail['Kategori'] == kategori]
        tolak = (sub['Keyakinan (%)'] < t * 100).mean() * 100
        r[kategori] = f"{tolak:.0f}%"
    baris.append(r)

df_sens = pd.DataFrame(baris)
print(df_sens.to_string(index=False))
df_sens.to_csv('hasil_ood_sensitivitas.csv', index=False)

print("\nCatatan: pada kategori 'kontrol_anjing', tingkat penolakan yang "
      "TINGGI justru menandakan ambang terlalu ketat (citra anjing asli "
      "ikut tertolak). Ambang optimal menyeimbangkan penolakan tinggi "
      "pada kategori OOD dan penolakan rendah pada kategori kontrol.")