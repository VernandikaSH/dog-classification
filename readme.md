# Perbandingan Performa MobileNetV2 dan EfficientNet-B0 pada Klasifikasi Citra Ras Anjing

Repositori ini berisi kode sumber dan implementasi sistem komparasi arsitektur Deep Learning (MobileNetV2 dan EfficientNet-B0) untuk klasifikasi citra halus (fine-grained visual recognition) pada 15 ras anjing pilihan. Model hasil pelatihan diintegrasikan ke dalam antarmuka aplikasi berbasis web menggunakan framework Streamlit.

* Nama: Vernandika Stanley Hansen
* NPM: 140810220031
* Program Studi: S-1 Teknik Informatika, Universitas Padjadjaran

## Deskripsi Proyek
Penelitian ini bertujuan untuk menganalisis trade-off antara akurasi prediksi dan efisiensi komputasi (waktu pelatihan) dari dua filosofi arsitektur lightweight CNN yang berbeda. Eksperimen dilakukan menggunakan teknik transfer learning (full layer freezing) dengan bobot pre-trained ImageNet. 

Sistem mengimplementasikan pendekatan Top-3 Probability untuk meminimalisir ambiguitas visual akibat inter-class similarity tinggi pada ras anjing serumpun atau fase anak anjing (puppy).

## Daftar 15 Kategori Ras Anjing Eksperimen
Eksperimen ini menggunakan subset data hasil kurasi manual dari Stanford Dogs Dataset yang mencakup 15 ras anjing berikut:
1. Beagle
2. Boxer
3. Chihuahua
4. Cocker Spaniel
5. Doberman
6. French Bulldog
7. German Shepherd
8. Golden Retriever
9. Labrador Retriever
10. Malamute
11. Pug
12. Rottweiler
13. Samoyed
14. Siberian Husky
15. Shih Tzu

## Metrik Performa Model (15 Kelas)
Berdasarkan hasil pengujian pada skenario 15 ras anjing (total 2.250 citra mentah dengan pembagian 80% training set dan 20% validation set), berikut adalah ringkasan performa akhir kedua model:

* MobileNetV2
  - Validation Accuracy (Top-1): 91.56%
  - Validation Accuracy (Top-3): 99.11%
  - Validation Loss: 0.2977
  - Total Waktu Pelatihan: 20 menit 17 detik

* EfficientNet-B0
  - Validation Accuracy (Top-1): 93.78%
  - Validation Accuracy (Top-3): 100.00%
  - Validation Loss: 0.1923
  - Total Waktu Pelatihan: 20 menit 29 detik

## Struktur Repositori
* app.py: Skrip utama untuk menjalankan aplikasi web Streamlit.
* requirements.txt: Daftar pustaka (dependencies) yang dibutuhkan oleh sistem.
* mobile.weights.h5: File parameter bobot hasil pelatihan model MobileNetV2.
* eff.weights.h5: File parameter bobot hasil pelatihan model EfficientNet-B0.

## Prasyarat Sistem
Pastikan perangkat Anda telah terinstal Python (disarankan versi 3.11 atau 3.12). Instalasi pustaka pendukung dapat dilakukan dengan mengeksekusi perintah berikut pada terminal:

```bash
pip install -r requirements.txt
