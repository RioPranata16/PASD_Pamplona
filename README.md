# JudolDetector Web API

Aplikasi berbasis antarmuka web dan REST API untuk mendeteksi serta mengklasifikasikan komentar promosi judi online (judol) secara otomatis menggunakan pemodelan *ensemble machine learning*.

Proyek ini dikembangkan oleh:
- Agung Widiyanto (2211110001)
- Ahnaf Ariacheda (103132400037)
- I Kadek Rio Adi Pranata Kusuma (103132400029)
- Raymundus Ariel Abas (103132430021)

## Fitur Utama
- **Prediksi Teks Tunggal:** Endpoint REST API untuk memproses satu kalimat dan mengembalikan probabilitas / label apakah teks tersebut merupakan promosi judi online.
- **Prediksi Batch (CSV):** Endpoint untuk menerima unggahan file CSV berisi banyak komentar, memprosesnya di memori (tanpa menyimpan file sampah di server), dan mengembalikan file CSV yang sudah diberi label prediksi.
- **Ekstraksi Fitur Teks:** Menggunakan TF-IDF Vectorizer (Unigram & Bigram, max_features=5000) yang sensitif terhadap pola bahasa gaul, singkatan, dan frasa promosi.
- **Multi-Model ML:** Terintegrasi dengan model CatBoost, XGBoost, dan LightGBM.

## Struktur Repositori
```text
PASD_Pamplona/
├── README.md
├── requirements.txt
├── .gitignore
└── deteksi-judol/
    ├── app.py                 # File utama untuk menjalankan server Flask
    ├── src/
    │   └── detector.py        # Core logic untuk pra-pemrosesan teks dan prediksi ML
    ├── models/                # Folder penyimpanan model (.pkl)
    ├── data/                  # Folder dataset (train, test, holdout)
    ├── templates/             # File HTML/CSS untuk antarmuka web
    └── notebooks/             # Eksperimen dan proses training model (Jupyter Notebook)
```

## Cara Instalasi dan Penggunaan

1. **Clone Repositori**
   ```bash
   git clone https://github.com/riopranata16/pasd_pamplona.git
   cd pasd_pamplona/deteksi-judol
   ```

2. **(Opsional) Buat Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   # venv\Scripts ctivate   # Untuk Windows
   ```

3. **Install Dependensi**
   ```bash
   pip install -r ../requirements.txt
   ```

4. **Jalankan Aplikasi**
   ```bash
   python app.py
   ```
   Aplikasi web akan berjalan secara lokal, biasanya di `http://127.0.0.1:5000/`.

## Endpoint API

- `POST /api/predict_text`
  - **Input:** JSON `{"text": "isi komentar"}`
  - **Output:** JSON berisi label prediksi ("Judol" / "Aman") dan *confidence score*.
- `POST /api/predict_csv`
  - **Input:** Form-data upload file `.csv`
  - **Output:** File `.csv` yang diunduh otomatis dengan tambahan kolom label prediksi.
