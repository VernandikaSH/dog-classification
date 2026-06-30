# Perbandingan Performa MobileNetV2 dan EfficientNet-B0 pada Klasifikasi Citra Ras Anjing

Repositori ini berisi kode sumber dan implementasi sistem komparasi arsitektur Deep Learning (MobileNetV2 dan EfficientNet-B0) untuk klasifikasi citra halus (fine-grained visual recognition) pada ras anjing. Model hasil pelatihan diintegrasikan ke dalam antarmuka aplikasi berbasis web menggunakan framework Streamlit.

* Nama: Vernandika Stanley Hansen
* NPM: 140810220031
* Program Studi: S-1 Teknik Informatika, Universitas Padjadjaran

## Deskripsi Proyek
Penelitian ini bertujuan untuk menganalisis trade-off antara akurasi prediksi dan efisiensi komputasi (waktu pelatihan) dari dua filosofi arsitektur lightweight CNN yang berbeda. Eksperimen dilakukan menggunakan teknik transfer learning (full layer freezing) dengan bobot pre-trained ImageNet. 

Sistem mengimplementasikan pendekatan Top-3 Probability untuk meminimalisir ambiguitas visual akibat inter-class similarity tinggi pada ras anjing serumpun atau fase anak anjing (puppy).

## Metrik Performa Model (120 Kelas)
Berdasarkan hasil pengujian skala penuh menggunakan Stanford Dogs Dataset (120 Kelas) selama 50 epoch, berikut adalah ringkasan performa kedua model:

* MobileNetV2
  - Validation Accuracy (Top-1): 74.37%
  - Validation Accuracy (Top-3): 91.92%
  - Total Waktu Pelatihan: 88 menit 27 detik

* EfficientNet-B0
  - Validation Accuracy (Top-1): 83.54%
  - Validation Accuracy (Top-3): 96.71%
  - Total Waktu Pelatihan: 89 menit 30 detik

## Struktur Repositori
* app.py: Skrip utama untuk menjalankan aplikasi web Streamlit.
* requirements.txt: Daftar pustaka (dependencies) yang dibutuhkan oleh sistem.
* mobile.weights.h5: File parameter bobot hasil pelatihan model MobileNetV2.
* eff.weights.h5: File parameter bobot hasil pelatihan model EfficientNet-B0.

## Prasyarat Sistem
Pastikan perangkat Anda telah terinstal Python (disarankan versi 3.11 atau 3.12). Instalasi pustaka pendukung dapat dilakukan dengan mengeksekusi perintah berikut pada terminal:

```bash
pip install -r requirements.txt
