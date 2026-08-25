import json
import os
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# Set page configuration
st.set_page_config(
    page_title="Penjaga Udara — Klasifikasi Kualitas Udara",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Konfigurasi & Load Model (Cached)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "model_xgb_kualitas_udara.json")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_mapping.json")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.json")

@st.cache_resource
def load_ml_assets():
    # Load XGBoost model
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Load label mapping
    with open(LABEL_MAP_PATH, "r") as f:
        _raw_label_mapping = json.load(f)
        label_mapping = {int(k): v for k, v in _raw_label_mapping.items()}
        
    # Load feature order
    with open(FEATURE_ORDER_PATH, "r") as f:
        feature_order = json.load(f)
        
    return model, label_mapping, feature_order

try:
    model, LABEL_MAPPING, FEATURE_ORDER = load_ml_assets()
except Exception as e:
    st.error(f"Gagal memuat model/metadata. Pastikan file model ada di folder `./model/`. Error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Data Rekomendasi & Deskripsi
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
        "warna": "#e65100",       # oranye/kuning tua
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
# Custom CSS untuk Desain Premium
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #fafafa;
    }
    
    h1, h2, h3, .title-font {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Topbar Header Style */
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        background: #ffffff;
        border-bottom: 1px solid #eaeaea;
        margin-bottom: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .topbar__brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .topbar__mark {
        width: 12px;
        height: 12px;
        background-color: #000;
        border-radius: 50%;
    }
    .topbar__name {
        font-weight: 700;
        font-size: 1.1rem;
        color: #111;
        letter-spacing: -0.02em;
    }
    .topbar__tag {
        font-size: 0.85rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Card Styles */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #eaeaea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.01);
        margin-bottom: 1.5rem;
    }
    
    /* Prediction Banner */
    .predict-banner {
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 5px solid;
    }
    
    /* Horizon Track style */
    .horizon-track {
        height: 8px;
        border-radius: 4px;
        display: flex;
        overflow: hidden;
        margin: 1.5rem 0 0.5rem 0;
        background: #eee;
    }
    .horizon-zone {
        flex: 1;
    }
    .hz-baik { background-color: #2e7d32; opacity: 0.3; }
    .hz-sedang { background-color: #1565c0; opacity: 0.3; }
    .hz-tidak-sehat { background-color: #e65100; opacity: 0.3; }
    
    .hz-baik.active { opacity: 1; }
    .hz-sedang.active { opacity: 1; }
    .hz-tidak-sehat.active { opacity: 1; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="topbar">
  <div class="topbar__brand">
    <span class="topbar__mark"></span>
    <span class="topbar__name">Penjaga Udara</span>
  </div>
  <span class="topbar__tag">DKI Jakarta &middot; Stasiun Bunderan HI &middot; XGBoost</span>
</div>
""", unsafe_allow_html=True)

# Layout: 2 Kolom
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<h2 class="title-font" style="margin-top: 0;">Masukkan Parameter Polutan</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #666; font-size: 0.95rem; margin-bottom: 1.5rem;">Isi konsentrasi enam polutan utama di bawah ini untuk mengklasifikasikan kualitas udara secara langsung.</p>', unsafe_allow_html=True)
    
    # Form input polutan
    inputs = {}
    
    # Grid inputs dengan custom styling
    with st.container():
        for feature in FEATURE_ORDER:
            # Tentukan unit
            unit = "mg/m³" if feature.lower() == "co" else "µg/m³"
            label = f"{feature.upper()} ({unit})"
            
            # Input angka
            inputs[feature] = st.number_input(
                label=label,
                min_value=0.0,
                value=0.0,
                step=0.1 if feature.lower() == "co" else 1.0,
                format="%.1f" if feature.lower() == "co" else "%.0f",
                key=f"input_{feature}"
            )

with col2:
    st.markdown('<h2 class="title-font" style="margin-top: 0;">Hasil Klasifikasi Kualitas Udara</h2>', unsafe_allow_html=True)
    
    # Cek apakah semua input masih 0
    all_zero = all(v == 0.0 for v in inputs.values())
    
    if all_zero:
        # Tampilan Empty State
        st.markdown("""
        <div class="custom-card" style="text-align: center; color: #777; padding: 4rem 2rem;">
            <svg viewBox="0 0 64 64" width="60" height="60" style="margin: 0 auto 1.5rem auto; stroke: #ccc; fill: none; stroke-width: 2.5; stroke-linecap: round;">
                <path d="M8 24c8-14 40-14 48 0M14 34c6-9 30-9 36 0M22 44c4-5 16-5 20 0" />
            </svg>
            <p style="font-size: 1rem; font-weight: 500;">Hasil klasifikasi akan muncul di sini secara real-time setelah Anda memasukkan data polutan.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Siapkan DataFrame input untuk model
        input_data = [inputs[feat] for feat in FEATURE_ORDER]
        input_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
        
        # Prediksi
        pred_encoded = int(model.predict(input_df)[0])
        pred_proba = model.predict_proba(input_df)[0]
        confidence = round(float(max(pred_proba)) * 100, 2)
        
        kategori = LABEL_MAPPING.get(pred_encoded, "UNKNOWN")
        info = REKOMENDASI.get(kategori, DEFAULT_INFO)
        
        # Banner Status Hasil
        st.markdown(f"""
        <div class="predict-banner" style="background-color: {info['warna_bg']}; border-left-color: {info['warna']}; color: {info['warna']};">
            <p style="margin: 0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Kategori Udara</p>
            <h1 class="title-font" style="margin: 0.2rem 0 0 0; font-size: 2.5rem; font-weight: 700; color: {info['warna']};">{kategori}</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; font-weight: 500;">Tingkat Kepercayaan Model: <strong>{confidence}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Deskripsi
        st.markdown(f"""
        <div class="custom-card">
            <h4 class="title-font" style="margin-top: 0; color: #333;">Deskripsi Kualitas</h4>
            <p style="color: #555; line-height: 1.6; font-size: 0.95rem;">{info['deskripsi']}</p>
            
            <div class="horizon-track">
                <div class="horizon-zone hz-baik {'active' if kategori == 'BAIK' else ''}"></div>
                <div class="horizon-zone hz-sedang {'active' if kategori == 'SEDANG' else ''}"></div>
                <div class="horizon-zone hz-tidak-sehat {'active' if kategori == 'TIDAK SEHAT' else ''}"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #888; font-weight: 600; margin-top: 0.25rem;">
                <span style="color: {'#2e7d32' if kategori == 'BAIK' else '#888'}">BAIK</span>
                <span style="color: {'#1565c0' if kategori == 'SEDANG' else '#888'}">SEDANG</span>
                <span style="color: {'#e65100' if kategori == 'TIDAK SEHAT' else '#888'}">TIDAK SEHAT</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Saran & Pencegahan
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="title-font" style="margin-top: 0; color: #333; margin-bottom: 1rem;">Saran & Pencegahan</h4>', unsafe_allow_html=True)
        for saran in info["saran"]:
            st.markdown(f"- {saran}")
        st.markdown('</div>', unsafe_allow_html=True)

# Footnote
st.markdown("""
---
<p style="text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem;">
    Prediksi bersifat estimatif berdasarkan data historis stasiun DKI1 (Bunderan HI) dan bukan pengganti informasi resmi ISPU dari KLHK / instansi terkait.
</p>
""", unsafe_allow_html=True)
