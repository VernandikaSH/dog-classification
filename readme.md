# Perbandingan Performa MobileNetV2 dan EfficientNet-B0 pada Klasifikasi Citra Ras Anjing

Repositori ini berisi kode sumber dan implementasi sistem komparasi arsitektur Deep Learning (MobileNetV2 dan EfficientNet-B0) untuk klasifikasi citra halus (fine-grained visual recognition) pada 15 ras anjing pilihan. Model hasil pelatihan diintegrasikan ke dalam antarmuka aplikasi berbasis web menggunakan framework Streamlit.

* Nama: Vernandika Stanley Hansen
* NPM: 140810220031
* Program Studi: S-1 Teknik Informatika, Universitas Padjadjaran

## Deskripsi Proyek

Penelitian ini menganalisis trade-off antara akurasi prediksi dan efisiensi komputasi dari dua filosofi arsitektur lightweight CNN yang berbeda. Eksperimen dilakukan menggunakan teknik transfer learning dengan strategi **fine-tuning dua tahap** menggunakan bobot pre-trained ImageNet, yaitu tahap pemanasan dengan convolutional base dibekukan, dilanjutkan tahap fine-tuning dengan membuka 30 lapisan terakhir.

Sistem mengimplementasikan pendekatan Top-3 Probability untuk meminimalisir ambiguitas visual akibat inter-class similarity tinggi pada ras anjing serumpun atau fase anak anjing (puppy).

## Daftar 15 Kategori Ras Anjing Eksperimen

Eksperimen menggunakan subset data hasil kurasi manual dari Stanford Dogs Dataset yang mencakup 15 ras anjing berikut:

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

## Konfigurasi Dataset

Total 2.593 citra mentah dibagi secara fisik dengan proporsi 60:20:20, yaitu 1.564 citra latih, 516 citra validasi, dan 513 citra uji. Testing set dipisahkan sejak awal dan tidak digunakan selama pelatihan maupun pemilihan model. Data latih diperkaya melalui augmentasi offline hingga setiap kelas berjumlah tepat 500 citra.

## Metrik Performa Model

Seluruh metrik berikut dilaporkan dari testing set independen sebanyak 513 citra.

| Metrik | MobileNetV2 | EfficientNet-B0 |
|---|---|---|
| Test Accuracy (Top-1) | 90,45% | 94,54% |
| Test Accuracy (Top-3) | 98,83% | 99,81% |
| Test Loss | 0,2612 | 0,1513 |
| Jumlah Parameter | 2.277.199 | 4.068.786 |
| Total Waktu Pelatihan | 6m 37s | 8m 11s |
| Waktu Inferensi (CPU lokal) | 92,5 ms | 133,1 ms |

Melalui pendekatan Top-3 Probability, 87,76% kesalahan prediksi pada MobileNetV2 dan 96,43% pada EfficientNet-B0 berhasil dipulihkan.

## Struktur Repositori

* `app.py` — Skrip utama aplikasi web Streamlit
* `tes_inference.py` — Skrip pengukuran waktu inferensi pada CPU lokal
* `uji_ood.py` — Skrip pengujian masukan di luar cakupan secara batch
* `dog-classification.ipynb` — Notebook pelatihan dan evaluasi model
* `requirements.txt` — Daftar pustaka yang dibutuhkan sistem
* `mobile.weights.h5` — Berkas bobot hasil pelatihan MobileNetV2
* `eff.weights.h5` — Berkas bobot hasil pelatihan EfficientNet-B0

## Prasyarat Sistem

Pastikan perangkat telah terinstal Python versi 3.11 atau 3.12. Instalasi pustaka pendukung dilakukan dengan perintah berikut:

```bash
pip install -r requirements.txt
```

Aplikasi memerlukan koneksi internet pada eksekusi pertama untuk mengunduh bobot pre-trained ImageNet, yang diperlukan sebagai kerangka inisialisasi arsitektur. Setelah tersimpan pada cache lokal, aplikasi dapat dijalankan secara luring.

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka pada web, kemudian pengguna dapat mengunggah citra anjing berformat JPG, JPEG, atau PNG untuk memperoleh tiga kandidat ras teratas dari kedua arsitektur.