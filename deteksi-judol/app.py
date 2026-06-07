from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import joblib
import re
import io

# ==========================================
# 1. CLASS UNTUK PREPROCESSING TEKS (OOP)
# ==========================================
class TextPreprocessor:
    # Pola Unicode untuk emoji (lebih presisi dari [^\w\s,] lama)
    EMOJI_PATTERN = re.compile(
        "["
        u"\U0001F600-\U0001F64F"   # emoticons wajah
        u"\U0001F300-\U0001F5FF"   # simbol & piktograf
        u"\U0001F680-\U0001F6FF"   # transportasi & peta
        u"\U0001F700-\U0001F77F"   # simbol alkimia
        u"\U0001F780-\U0001F7FF"   # geometri tambahan
        u"\U0001F800-\U0001F8FF"   # simbol suplemen
        u"\U0001F900-\U0001F9FF"   # simbol suplemen tambahan
        u"\U0001FA00-\U0001FA6F"   # simbol catur
        u"\U0001FA70-\U0001FAFF"   # simbol & piktograf lanjutan
        u"\U00002702-\U000027B0"   # dingbat
        u"\U000024C2-\U0001F251"   # simbol terlampir
        "]+",
        flags=re.UNICODE,
    )

    def __init__(self):
        self.stopwords = {'yang', 'di', 'ke', 'dari', 'pada', 'dan', 'atau', 'ini', 'itu', 'yg', 'dgn'}

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Ekstrak emoji sebelum teks dinormalisasi
        emojis = ' '.join(self.EMOJI_PATTERN.findall(text))
        text_lower = text.lower()
        # Hapus karakter selain huruf, angka, dan spasi
        text_clean = re.sub(r'[^a-z0-9\s]', ' ', text_lower)
        words = text_clean.split()
        filtered = [w for w in words if w and w not in self.stopwords]
        return ' '.join(filtered) + (' ' + emojis if emojis else '')


# ==========================================
# 2. CLASS UNTUK MODEL MACHINE LEARNING (OOP)
# ==========================================
class JudolModel:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.models: dict = {}
        self.load_models()

    def load_models(self):
        try:
            self.vectorizer = joblib.load("tfidf_vectorizer_full.pkl")
            self.models = {
                "XGBoost":  joblib.load("xgboost_full.pkl"),
                "LightGBM": joblib.load("lightgbm_full.pkl"),
                "CatBoost": joblib.load("catboost_full.pkl"),
            }
            print("✅ Semua model berhasil dimuat!")
        except Exception as e:
            print(f"❌ Gagal memuat model: {e}")

    def predict(self, texts: list[str], model_name: str = "LightGBM") -> list[dict]:
        """
        Prediksi dengan confidence score via predict_proba.
        Fallback ke predict() jika model tidak mendukung proba.
        """
        if self.vectorizer is None or model_name not in self.models:
            raise ValueError("Model belum dimuat atau nama model tidak valid.")

        cleaned = [self.preprocessor.clean_text(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        model = self.models[model_name]

        # Ambil confidence score (probabilitas kelas Judol = indeks 1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            confidences = proba[:, 1].tolist()          # P(Judol)
            predictions = [1 if p >= 0.5 else 0 for p in confidences]
        else:
            # Fallback: tanpa proba (misal SVM tanpa probability=True)
            predictions = model.predict(X).tolist()
            confidences = [1.0 if p == 1 else 0.0 for p in predictions]

        results = []
        for text, pred, conf in zip(texts, predictions, confidences):
            is_judol = pred == 1
            results.append({
                "teks_asli":  text,
                "prediksi":   "Judol" if is_judol else "Bukan Judol",
                "status":     "danger" if is_judol else "success",
                "confidence": round(conf * 100, 1),   # Persen, 1 desimal
            })
        return results


# ==========================================
# 3. FLASK APP (CONTROLLER)
# ==========================================
app = Flask(__name__)
judol_classifier = JudolModel()

# Default model diubah ke LightGBM (akurasi terbaik di holdout)
DEFAULT_MODEL = "LightGBM"

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict_text", methods=["POST"])
def predict_text():
    """Endpoint prediksi teks manual."""
    data = request.json
    text_input = data.get("text", "")
    model_name = data.get("model", DEFAULT_MODEL)

    if not text_input.strip():
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    texts = [t.strip() for t in text_input.split('\n') if t.strip()]
    try:
        results = judol_classifier.predict(texts, model_name)
        n_judol = sum(1 for r in results if r["status"] == "danger")
        n_aman  = len(results) - n_judol
        return jsonify({
            "results": results,
            "stats": {"total": len(results), "judol": n_judol, "aman": n_aman},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_csv", methods=["POST"])
def predict_csv():
    """Endpoint prediksi massal via CSV."""
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    file = request.files['file']
    model_name = request.form.get("model", DEFAULT_MODEL)

    if file.filename == '':
        return jsonify({"error": "File belum dipilih"}), 400

    try:
        df = pd.read_csv(file, sep=None, engine='python')

        # Deteksi kolom teks
        text_column = None
        for col_hint in ['comment', 'teks', 'text', 'komentar']:
            matches = [c for c in df.columns if c.lower() == col_hint]
            if matches:
                text_column = matches[0]
                break
        if text_column is None:
            text_column = df.columns[0]

        texts = df[text_column].astype(str).tolist()
        results = judol_classifier.predict(texts, model_name)

        df['Prediksi_Label']      = [r['prediksi']   for r in results]
        df['Confidence_Judol_Pct'] = [r['confidence'] for r in results]

        output = io.StringIO()
        df.to_csv(output, index=False, sep=';')
        output.seek(0)

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=hasil_{file.filename}"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8080)