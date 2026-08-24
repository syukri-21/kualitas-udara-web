"""
Backend Flask untuk Klasifikasi Kualitas Udara (XGBoost)
==========================================================
Menyediakan endpoint:
  - GET  /            -> halaman web (form input polutan)
  - POST /predict      -> menerima 6 nilai polutan (JSON), mengembalikan
                           kategori + deskripsi + saran pencegahan

Sebelum menjalankan, pastikan 3 file hasil training dari Google Colab
sudah diletakkan di folder ./model/:
  - model_xgb_kualitas_udara.json
  - label_mapping.json
  - feature_order.json
"""

import json
import os

from flask import Flask, jsonify, render_template, request
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Konfigurasi & load artefak model
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "model_xgb_kualitas_udara.json")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_mapping.json")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.json")

app = Flask(__name__)

# Load model XGBoost (format native .json yang disimpan lewat model.save_model())
model = XGBClassifier()
model.load_model(MODEL_PATH)

# Load mapping label: {"0": "BAIK", "1": "SEDANG", "2": "TIDAK SEHAT"}
with open(LABEL_MAP_PATH, "r") as f:
    _raw_label_mapping = json.load(f)
    # pastikan key berupa int agar gampang dipetakan dari hasil predict()
    LABEL_MAPPING = {int(k): v for k, v in _raw_label_mapping.items()}

# Load urutan fitur, WAJIB sama persis dengan urutan saat training
with open(FEATURE_ORDER_PATH, "r") as f:
    FEATURE_ORDER = json.load(f)  # contoh: ["pm25", "pm10", "so2", "co", "o3", "no2"]

# ---------------------------------------------------------------------------
# Rekomendasi / deskripsi per kategori (samakan dengan yang ada di notebook)
# ---------------------------------------------------------------------------
REKOMENDASI = {
    "BAIK": {
        "deskripsi": "Kualitas udara sangat baik dan tidak memberikan efek negatif terhadap manusia, hewan, maupun tumbuhan.",
        "saran": [
            "Kualitas udara aman untuk seluruh aktivitas luar ruangan.",
            "Tetap jaga kualitas udara dengan mengurangi penggunaan kendaraan pribadi dan pembakaran sampah.",
            "Waktu yang baik untuk berolahraga di luar ruangan.",
        ],
        "warna": "#2e7d32",       # hijau
        "warna_bg": "#e8f5e9",
    },
    "SEDANG": {
        "deskripsi": "Kualitas udara masih dapat diterima, namun bisa berdampak ringan pada kelompok yang sensitif terhadap polusi (misalnya penderita asma atau gangguan pernapasan).",
        "saran": [
            "Kelompok sensitif (anak-anak, lansia, ibu hamil, penderita gangguan pernapasan) disarankan mengurangi aktivitas fisik berat yang lama di luar ruangan.",
            "Gunakan masker jika beraktivitas di luar dalam waktu lama, terutama di area padat kendaraan.",
            "Pastikan ventilasi rumah tetap baik namun hindari jam-jam puncak polusi (pagi & sore hari padat kendaraan).",
        ],
        "warna": "#1565c0",       # biru
        "warna_bg": "#e3f2fd",
    },
    "TIDAK SEHAT": {
        "deskripsi": "Kualitas udara dapat merugikan kesehatan manusia maupun hewan, serta menimbulkan efek pada tumbuhan dan nilai estetika.",
        "saran": [
            "Kurangi aktivitas fisik berat di luar ruangan, khususnya bagi kelompok sensitif.",
            "Gunakan masker (idealnya masker N95) saat berada di luar ruangan.",
            "Tutup jendela/pintu saat polusi tinggi dan gunakan air purifier bila tersedia.",
            "Segera periksakan diri ke fasilitas kesehatan jika muncul gejala pernapasan seperti sesak napas atau batuk berkepanjangan.",
        ],
        "warna": "#e65100",       # oranye/kuning tua (mewakili "tidak sehat")
        "warna_bg": "#fff3e0",
    },
}

DEFAULT_INFO = {
    "deskripsi": "Kategori tidak dikenali.",
    "saran": [],
    "warna": "#616161",
    "warna_bg": "#f5f5f5",
}


# ---------------------------------------------------------------------------
# Helper: validasi & parsing input
# ---------------------------------------------------------------------------
def parse_input(payload: dict):
    """Ambil & validasi 6 nilai polutan dari request JSON sesuai FEATURE_ORDER.

    Mengembalikan (values, error). Jika error is not None, values tidak valid.
    """
    values = []
    for key in FEATURE_ORDER:
        if key not in payload:
            return None, f"Field '{key}' tidak ditemukan pada request."
        try:
            val = float(payload[key])
        except (TypeError, ValueError):
            return None, f"Nilai '{key}' harus berupa angka."
        if val < 0:
            return None, f"Nilai '{key}' tidak boleh negatif."
        values.append(val)
    return values, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    # Kirim urutan fitur ke frontend supaya form otomatis mengikuti urutan
    # kolom yang dipakai model (tanpa perlu hardcode di HTML).
    return render_template("index.html", features=FEATURE_ORDER)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Body request harus berupa JSON."}), 400

    values, error = parse_input(payload)
    if error:
        return jsonify({"error": error}), 400

    # Susun DataFrame 1 baris sesuai urutan fitur training
    import pandas as pd

    input_df = pd.DataFrame([values], columns=FEATURE_ORDER)

    pred_encoded = int(model.predict(input_df)[0])
    pred_proba = model.predict_proba(input_df)[0]

    kategori = LABEL_MAPPING.get(pred_encoded, "UNKNOWN")
    info = REKOMENDASI.get(kategori, DEFAULT_INFO)

    response = {
        "kategori": kategori,
        "confidence": round(float(max(pred_proba)) * 100, 2),
        "deskripsi": info["deskripsi"],
        "saran": info["saran"],
        "warna": info["warna"],
        "warna_bg": info["warna_bg"],
        "input": dict(zip(FEATURE_ORDER, values)),
    }
    return jsonify(response)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": FEATURE_ORDER, "labels": LABEL_MAPPING})


if __name__ == "__main__":
    # debug=True hanya untuk pengembangan lokal, matikan saat deploy produksi
    app.run(host="0.0.0.0", port=5000, debug=True)
