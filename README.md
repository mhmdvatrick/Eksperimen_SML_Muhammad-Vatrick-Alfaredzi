# 🚀 Proyek Akhir Machine Learning: Telco Customer Churn

**Oleh:** Muhammad Vatrick Alfaredzi  
**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Proyek ini adalah implementasi *End-to-End Machine Learning Pipeline* untuk memprediksi pelanggan *churn* (berhenti berlangganan) pada perusahaan telekomunikasi. Proyek ini dibangun untuk memenuhi 4 Kriteria Utama (Level Advanced) dari course Dicoding.

---

## 🛠️ Teknologi yang Digunakan
- **Eksperimen & Preprocessing**: Python, Pandas, Scikit-Learn, Imbalanced-Learn (SMOTE), Jupyter Notebook
- **Model Tracking & Management**: MLflow, DagsHub
- **Continuous Integration (CI)**: GitHub Actions, MLflow Project, Docker Hub
- **Monitoring & Alerting**: Flask (Model Serving), Prometheus, Grafana

---

## 📁 Struktur Direktori & Fitur Utama

### 1. Eksperimen Data (`Eksperimen_SML_Muhammad-Vatrick-Alfaredzi.txt`)
- Meliputi Eksperimen Data awal (*Exploratory Data Analysis*) dalam bentuk Notebook.
- Otomatisasi pemrosesan data (Missing Values, Encoding, Scaling, SMOTE).
- Diotomatisasi menggunakan **GitHub Actions** untuk memproses data baru setiap ada perubahan.

### 2. Membangun Model (`Membangun_model/`)
- Membangun model **Random Forest Classifier**.
- Melakukan *Hyperparameter Tuning* (GridSearchCV).
- Menyimpan parameter, metrik, dan *artefak* secara remote menggunakan **MLflow & DagsHub**.
- Metrik tambahan: Confusion Matrix, Classification Report.

### 3. Workflow CI (`Workflow-CI.txt`)
- Menggunakan struktur standar **MLProject**.
- Setup CI menggunakan **GitHub Actions**.
- Ketika kode di-push, sistem otomatis menjalankan training MLflow, membangun Docker Image, lalu melakukan push Image tersebut ke **Docker Hub**.

### 4. Monitoring dan Logging (`Monitoring dan Logging/`)
- **Model Serving**: API menggunakan Flask untuk menerima data *inference* secara real-time.
- **Prometheus**: Melakukan scraping 10+ metrik model secara spesifik (Request rate, Latency p99, Error Rate, Memory/CPU Usage, Drift score, dll).
- **Grafana**: Visualisasi *Dashboard Monitoring* lengkap yang terhubung ke Prometheus.
- **Alerting**: Konfigurasi 3 peringatan otomatis (Latency > 1s, Error Rate > 5%, dan Prediksi *Drift*).

---

## 🏃 Cara Menjalankan Secara Lokal

### Kriteria 2 (Training Model)
Pastikan Anda berada di direktori `Membangun_model/` dan sudah menaruh file dataset preprocessed `telco_churn_clean.csv`.
```bash
pip install -r requirements.txt
python modelling.py
python modelling_tuning.py
```

### Kriteria 4 (Monitoring Stack)
Pastikan Anda sudah meng-install Docker dan Docker Compose.
1. Jalankan aplikasi *Serving Model*:
   ```bash
   cd "Monitoring dan Logging"
   pip install -r requirements.txt
   python model_serving.py
   ```
2. Jalankan Container Prometheus dan Grafana:
   ```bash
   docker-compose up -d
   ```
3. Lakukan simulasi *Load Test* untuk mengisi data di grafik:
   ```bash
   python 7.Inference.py
   ```
4. Buka URL berikut:
   - **Prometheus**: `http://localhost:9090`
   - **Grafana**: `http://localhost:3000` (User: admin | Pass: admin)

---

> *Proyek ini diajukan untuk mendapatkan skor kelulusan maksimal (Bintang 5 / 4 Points di semua Kriteria).*
