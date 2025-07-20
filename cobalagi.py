import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# === PERINTAH PERTAMA STREAMLIT: KONFIGURASI HALAMAN ===
# Dipindahkan ke sini agar menjadi perintah pertama yang dieksekusi
st.set_page_config(page_title="Analisis Sentimen", page_icon="📊", layout="wide")

# --- Konfigurasi Awal & Download Resource ---
# Download NLTK resource (jika belum ada)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Konfigurasi dasar
max_len = 30
label_to_id = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
id_to_label = {v: k for k, v in label_to_id.items()}

# --- Fungsi Pemuatan Aset dengan Caching ---
@st.cache_resource
def load_assets():
    """Memuat semua aset yang diperlukan seperti tokenizer, stopwords, slang, lexicon, dan stemmer."""
    assets = {
        "tokenizer": None, "stopwords": set(), "slang_dict": {},
        "lexicon": {}, "stemmer": None
    }
    
    # 1. Muat Stemmer
    factory = StemmerFactory()
    assets["stemmer"] = factory.create_stemmer()

    # 2. Muat Tokenizer
    tokenizer_path = 'models/tokenizer.pkl'
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            assets["tokenizer"] = pickle.load(handle)
    else:
        st.error(f"File krusial 'models/tokenizer.pkl' tidak ditemukan. Aplikasi tidak dapat berjalan.")
        st.stop()

    # 3. Muat Stopwords
    try:
        with open('assets/indonesian-stopwords-complete.txt', 'r', encoding='utf-8') as f:
            assets["stopwords"] = set(f.read().splitlines())
    except Exception as e:
        st.warning(f"Gagal memuat stopwords: {e}")

    # 4. Muat Kamus Slang
    slang_sources = [
        'https://github.com/adeariniputri/text-preprocesing/raw/master/slang.csv',
        'assets/slang.csv'
    ]
    df_slang_1 = pd.DataFrame()
    try:
        df_slang_1 = pd.read_csv(slang_sources[0])
    except:
        try:
            df_slang_1 = pd.read_csv(slang_sources[1])
        except Exception as e:
            st.warning(f"Gagal memuat kamus slang (sumber 1): {e}")
            
    if not df_slang_1.empty:
        for _, row in df_slang_1.iterrows():
            assets["slang_dict"][str(row.get('slang', '')).strip()] = str(row.get('formal', '')).strip()

    return assets

@st.cache_resource
def load_all_models(model_dir='models'):
    """Memuat semua model .h5 dari direktori yang ditentukan."""
    models = {}
    if not os.path.exists(model_dir):
        st.error(f"Direktori model '{model_dir}' tidak ditemukan.")
        return models

    for root, _, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith('.h5') and filename.startswith("best_model"):
                relative_path = os.path.relpath(os.path.join(root, filename), model_dir)
                model_name = relative_path.replace(".h5", "").replace("best_model_", "").replace("/", " - ")
                model_path = os.path.join(root, filename)
                try:
                    models[model_name] = tf.keras.models.load_model(model_path, compile=False)
                except Exception as e:
                    st.warning(f"Gagal memuat model {model_name}: {e}")
    return models

# --- Fungsi Preprocessing ---
def preprocess_text(text, assets):
    """Fungsi preprocessing teks lengkap."""
    stemmer = assets["stemmer"]
    stopwords = assets["stopwords"]
    slang_dict = assets["slang_dict"]
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'rt[\s]+', '', text)
    text = re.sub(r'[^a-z\s-]', '', text)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    tokens = nltk.tokenize.word_tokenize(text)
    processed_tokens = [slang_dict.get(token, token) for token in tokens if slang_dict.get(token, token) not in stopwords]
    stemmed_tokens = [stemmer.stem(word) for word in processed_tokens]
    
    return ' '.join(stemmed_tokens)

# --- Memuat Aset dan Model (dengan feedback di UI) ---
with st.spinner("Mempersiapkan aset (pertama kali mungkin butuh waktu)..."):
    loaded_assets = load_assets()
    
with st.spinner("Mempersiapkan model-model AI..."):
    all_models = load_all_models()

# === TAMPILAN UTAMA APLIKASI GUI ===
st.title("📊 Analisis Sentimen Multi-Model")
st.markdown("Aplikasi ini menggunakan beberapa model Deep Learning untuk menganalisis sentimen dari sebuah teks ulasan (Positif, Negatif, atau Netral).")

st.markdown("---")

# Area Input Pengguna
user_input = st.text_area("Masukkan teks ulasan atau komentar Anda di sini:", height=150, placeholder="Contoh: Aplikasi ini luar biasa, fiturnya sangat membantu!")

if st.button("Analisis Sentimen", type="primary"):
    if user_input.strip() and all_models:
        with st.spinner("Sedang menganalisis..."):
            cleaned_text = preprocess_text(user_input, loaded_assets)
            tokenizer = loaded_assets["tokenizer"]
            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

            predictions = []
            for model_name, model_obj in all_models.items():
                prediction = model_obj.predict(padded_sequence, verbose=0)
                predicted_class_id = np.argmax(prediction, axis=1)[0]
                
                predictions.append({
                    'model_name': model_name,
                    'predicted_sentiment': id_to_label.get(predicted_class_id, "Tidak Dikenal"),
                    'confidence': np.max(prediction)
                })
            
            sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

        st.markdown("---")
        st.subheader("🔬 Hasil Analisis")

        col1, col2 = st.columns(2)
        with col1:
            st.info("**Teks Asli Anda:**")
            st.write(f"_{user_input}_")
        with col2:
            st.info("**Teks Setelah Diproses AI:**")
            st.write(f"_{cleaned_text}_")

        st.markdown("<br>", unsafe_allow_html=True)
        st.success("**Prediksi dari Setiap Model:**")
        
        sentiment_emoji_map = {'Positif': '😊', 'Negatif': '😠', 'Netral': '😐'}

        cols = st.columns(len(sorted_predictions))
        for i, res in enumerate(sorted_predictions):
            with cols[i]:
                st.metric(
                    label=f"📦 Model: {res['model_name']}", 
                    value=f"{sentiment_emoji_map.get(res['predicted_sentiment'], '❓')} {res['predicted_sentiment']}",
                    delta=f"Kepercayaan: {res['confidence']:.2%}",
                    delta_color="off"
                )

    elif not all_models:
        st.error("Tidak ada model yang berhasil dimuat. Aplikasi tidak dapat melakukan prediksi.")
    else:
        st.warning("Mohon masukkan teks terlebih dahulu untuk dianalisis.")

# Footer
st.markdown("---")
st.write("Dibuat dengan Streamlit dan TensorFlow")