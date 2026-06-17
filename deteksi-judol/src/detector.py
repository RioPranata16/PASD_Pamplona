# detector.py
import os
import re
import joblib

class JudolDetector:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        # Gunakan list stopwords yang lengkap di sini
        self.stopwords = set([
            'yang','dan','di','ke','dari','ini','itu','dengan','untuk','adalah','akan',
            'ada','juga','tidak','pada','karena','kita','dalam','saya','anda','sudah',
            'bisa','lebih','kamu','mereka','jika','maka','atau','tetapi','namun','pun',
            'apa','semua','banyak','ketika','setelah','sebelum','lagi','bila','bagaimana',
            'mengapa','siapa','dimana','kapan','apakah','sedang','telah','pernah','hanya',
            'seperti','kami','belum','kalau','sebuah','sebagai','oleh','tapi','hingga',
            'sampai','yaitu','agar','supaya','meski','walaupun','sekali','sangat','terlalu',
            'masih','selalu','sering','kadang','jarang','baik','buruk','besar','kecil',
            'baru','lama','paling','amat','nya','kan','lah','kah','pun','si','sang',
            'para','sih','nih','deh','dong','lo','lu','gue','gw','emang',
            'banget','udah','aja','doang','saja','mah','nah','ya','yuk',
            'dulu','nanti','tadi','barusan','sekarang','hari','malam','pagi','sore',
            'sama','buat','soal','hal','cara','tempat','waktu','orang'
        ])
        self._load_resources()

    def _load_resources(self):
        tfidf_path = os.path.join(self.base_dir, "tfidf_vectorizer_full.pkl")
        self.vectorizer = joblib.load(tfidf_path)
        
        self.models = {
            "XGBoost":  joblib.load(os.path.join(self.base_dir, "xgboost_full.pkl")),
            "LightGBM": joblib.load(os.path.join(self.base_dir, "lightgbm_full.pkl")),
            "CatBoost": joblib.load(os.path.join(self.base_dir, "catboost_full.pkl")),
        }

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        
        # Regex Unicode Emoji yang rapi
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF" u"\U0001F700-\U0001F77F"
            u"\U0001F780-\U0001F7FF" u"\U0001F800-\U0001F8FF"
            u"\U0001F900-\U0001F9FF" u"\U0001FA00-\U0001FA6F"
            u"\U0001FA70-\U0001FAFF" u"\u2600-\u26FF" u"\u2700-\u27BF"
            "]+", flags=re.UNICODE
        )
        emojis = emoji_pattern.findall(text)
        
        # Standarisasi pembersihan teks alfanumerik biasa
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = text.strip() + ' ' + ' '.join(emojis)
        
        tokens = [w for w in text.split() if w and w not in self.stopwords]
        return ' '.join(tokens)

    def predict(self, texts: list[str], model_name: str = "LightGBM") -> list[dict]:
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' tidak valid.")
            
        processed = [self.preprocess_text(t) for t in texts]
        
        # 1. Transformasi teks ke matriks TF-IDF
        X_sparse = self.vectorizer.transform(processed)
        
        # 2. BUNGKUS KE DATAFRAME + Beri nama fitur agar LightGBM/XGBoost tidak protes
        feature_names = self.vectorizer.get_feature_names_out()
        import pandas as pd
        X_df = pd.DataFrame(X_sparse.toarray(), columns=feature_names)
        
        model = self.models[model_name]
        
        # 3. Ambil confidence score menggunakan X_df
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_df)
            confidences = proba[:, 1].tolist()
            predictions = [1 if p >= 0.5 else 0 for p in confidences]
        else:
            predictions = model.predict(X_df).tolist()
            confidences = [1.0 if p == 1 else 0.0 for p in predictions]
            
        results = []
        for text, pred, conf in zip(texts, predictions, confidences):
            is_judol = pred == 1
            results.append({
                "teks_asli":  text,
                "prediksi":   "Judol" if is_judol else "Bukan Judol",
                "status":     "danger" if is_judol else "success",
                "confidence": round(conf * 100, 1),
            })
        return results