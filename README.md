# Deteksi Anomali pada Panel Surya Menggunakan Kombinasi Ekstraksi Fitur CNN dan Klasifikasi Ensemble

**Proyek Pembelajaran Mesin — Kelompok 11**  
S1 Sains Data, Fakultas Matematika dan IPA  
Universitas Negeri Surabaya | Kelas 2024-C

## Anggota Kelompok

| Nama | NIM |
|------|-----|
| M Shidqi Sharim Arrasyid (Ketua) | 24031554201 |
| M Nauval Sayyid Abdillah | 24031554092 |
| Arya Abdi Wicaksana | 24031554224 |

## Latar Belakang

Ekspansi elektrifikasi di daerah tertinggal, terdepan, dan terluar (3T) di Indonesia masih menghadapi tantangan besar dalam hal pemeliharaan infrastruktur energi terbarukan seperti panel surya. Minimnya pemanfaatan teknologi IoT dan AI menyebabkan anomali pada panel surya seringkali tidak terdeteksi hingga menyebabkan kerusakan fatal. Proyek ini mengembangkan sistem deteksi anomali otomatis menggunakan kombinasi ekstraksi fitur CNN dan klasifikasi ensemble untuk mendukung pemeliharaan berbasis AI yang efisien.

## Tujuan

- Mengembangkan sistem ekstraksi fitur otomatis dari citra panel surya menggunakan CNN
- Membangun model klasifikasi ensemble untuk mendeteksi dan mengklasifikasikan jenis anomali
- Mengevaluasi kinerja model gabungan (CNN + Ensemble) dalam mengklasifikasikan berbagai jenis anomali
- Memberikan solusi pemeliharaan preventif berbasis AI untuk sistem energi terbarukan di wilayah 3T

## Dataset

- **Sumber:** [Kaggle - Solar Panel Images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images/data)
- **Total:** 875 gambar
- **Kelas:** Clean, Bird-drop, Dusty, Electrical-damage, Physical-Damage, Snow-Covered
- **Split:** 80% train, 10% validation, 10% test
- **Catatan:** Terdapat sedikit ketidakseimbangan jumlah gambar antar kelas

## Alur Pengerjaan

1. Download dan eksplorasi dataset (EDA)
2. Cek distribusi kelas
3. Split data train/validation/test
4. Resize, normalisasi, dan augmentasi gambar
5. Load pre-trained CNN model (VGG16)
6. Ekstraksi fitur gambar menggunakan CNN
7. Simpan vektor fitur hasil ekstraksi sebagai input ensemble
8. Evaluasi model (Classification Report + F1 Score)
9. Visualisasi grafik training vs validation
10. Hyperparameter tuning

## Metode

### Ekstraksi Fitur — CNN Transfer Learning

- **MobileNetV2** — pre-trained ImageNet, base model frozen, custom classifier head (Dense 128 → Dropout 0.5 → Dense 6 softmax)
- **VGG16** — pre-trained ImageNet, digunakan untuk mengekstrak vektor fitur 25088 dimensi dari setiap citra

### Klasifikasi Ensemble (Rencana)

Fitur hasil ekstraksi VGG16 akan digunakan sebagai input untuk classifier ensemble (Random Forest, SVM, XGBoost) dengan teknik Soft Voting atau Stacking.

## Metrik Evaluasi

- Akurasi
- Presisi
- Recall
- F1-Score
- Confusion Matrix

## Tools & Library

- Python
- TensorFlow / Keras
- scikit-learn
- Matplotlib, Seaborn
- NumPy
- splitfolders

## Cara Menjalankan

```bash
pip install tensorflow scikit-learn matplotlib seaborn splitfolders kaggle
jupyter notebook "Progres_ml_kelompok_11_executed.ipynb"
```

## Struktur File

```
├── Progres_ml_kelompok_11.ipynb   # Notebook utama (sudah dieksekusi)
├── fitur_cnn.npy                           # Hasil ekstraksi fitur VGG16
├── dataset_split/                          # Data train/val/test
└── solar-panel-images/                     # Dataset mentah
```

## Referensi

1. CNN-based automatic detection of photovoltaic solar module anomalies in infrared images: a comparative study
2. Remote anomaly detection and classification of solar photovoltaic modules based on deep neural network
3. Deep Neural Network and Feature Extraction Approach for Anomaly Detection and Classification in Infrared Images of Photovoltaic Systems
4. Solar photovoltaic panel cells defects classification using deep learning ensemble methods
5. Enhanced CNN-LSTM Feature Extraction and Ensemble Learning for Anomaly Detection in Photovoltaic Data
