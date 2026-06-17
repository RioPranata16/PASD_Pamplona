# app.py
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import io
from src.detector import JudolDetector  # Mengambil core logika dari detector.py

app = Flask(__name__)

# Inisialisasi object detector sekali di awal aplikasi
import os
# Mengarahkan base_dir ke folder 'models'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

judol_classifier = JudolDetector(base_dir=MODEL_DIR)
DEFAULT_MODEL = "XGBoost"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict_text", methods=["POST"])
def predict_text():
    data = request.json
    text_input = data.get("text", "")
    model_name = data.get("model", DEFAULT_MODEL)

    if not text_input.strip():
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    texts = [t.strip() for t in text_input.split('\n') if t.strip()]
    try:
        results = judol_classifier.predict(texts, model_name)
        n_judol = sum(1 for r in results if r["status"] == "danger")
        return jsonify({
            "results": results,
            "stats": {"total": len(results), "judol": n_judol, "aman": len(results) - n_judol},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_csv", methods=["POST"])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    file = request.files['file']
    model_name = request.form.get("model", DEFAULT_MODEL)

    if file.filename == '':
        return jsonify({"error": "File belum dipilih"}), 400

    try:
        df = pd.read_csv(file, sep=None, engine='python')

        # Deteksi kolom teks otomatis
        text_column = None
        for col_hint in ['comment', 'teks', 'text', 'komentar']:
            matches = [c for c in df.columns if c.lower() == col_hint]
            if matches:
                text_column = matches[0]
                break
        if text_column is None:
            text_column = df.columns[0]

        texts = df[text_column].astype(str).tolist()
        
        # Panggil fungsi predict yang sudah rapi mengembalikan format dictionary
        results = judol_classifier.predict(texts, model_name)

        df['Prediksi_Label']       = [r['prediksi']   for r in results]
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