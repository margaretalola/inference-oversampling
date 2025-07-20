import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go
import pandas as pd
from nltk.tokenize import word_tokenize
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import requests
import string

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Sentimen Bahasa Indonesia",
    page_icon="🔍",
    layout="wide"
)

# --- Download Resource NLTK ---
@st.cache_resource
def download_nltk_resource():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Mendownload resource NLTK 'punkt'...")
        nltk.download('punkt')

download_nltk_resource()

# --- Fungsi Pemuatan Aset dengan Caching ---
# Menggunakan `st.cache_resource` agar aset hanya dimuat sekali saat aplikasi pertama kali dijalankan.
@st.cache_resource
def load_all_assets():
    """
    Memuat aset dengan pendekatan HIBRIDA:
    1. Mencoba dari URL online terlebih dahulu.
    2. Jika gagal, beralih ke file cadangan LOKAL.
    """
    assets = {
        "tokenizer": None, "stopwords": set(), "slang_dict": {},
        "lexicon": {}, "model_scores": {}, "stemmer": None
    }
    
    # 1. Muat Stemmer
    try:
        factory = StemmerFactory()
        assets["stemmer"] = factory.create_stemmer()
    except Exception as e:
        st.error(f"Kritis: Gagal inisialisasi Stemmer. Aplikasi tidak dapat berjalan dengan optimal. {e}")

    # 2. Muat Tokenizer
    tokenizer_path = 'models/tokenizer.pkl'
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            assets["tokenizer"] = pickle.load(handle)
    else:
        st.error(f"Kritis: File '{tokenizer_path}' tidak ditemukan. Aplikasi tidak dapat melakukan prediksi.")
        st.stop() # Hentikan eksekusi jika aset kritis tidak ditemukan

    # 3. Muat Stopwords (Lokal)
    try:
        with open('assets/indonesian-stopwords-complete.txt', 'r', encoding='utf-8') as f:
            assets["stopwords"] = set(f.read().splitlines())
    except Exception as e:
        st.warning(f"Gagal memuat stopwords dari file lokal: {e}")

    # 4. Muat Kamus Slang (Hibrida)
    slang_dict = {}
    slang_sources = [
        {'url': 'https://github.com/adeariniputri/text-preprocesing/raw/master/slang.csv', 'local': 'assets/slang.csv'},
        {'url': 'https://github.com/louisowen6/NLP_bahasa_resources/raw/master/combined_slang_words.txt', 'local': 'assets/combined_slang_words.txt', 'sep': ':'}
    ]
    for source in slang_sources:
        try:
            if 'sep' in source:
                df_slang = pd.read_csv(source['url'], sep=source['sep'], header=None, names=["slang", "formal"])
            else:
                df_slang = pd.read_csv(source['url'])
            for _, row in df_slang.iterrows():
                slang_dict[str(row.get('slang', '')).strip()] = str(row.get('formal', '')).strip()
        except Exception as e:
            try:
                if 'sep' in source:
                    df_slang = pd.read_csv(source['local'], sep=source['sep'], header=None, names=["slang", "formal"])
                else:
                    df_slang = pd.read_csv(source['local'])
                for _, row in df_slang.iterrows():
                    slang_dict[str(row.get('slang', '')).strip()] = str(row.get('formal', '')).strip()
            except Exception:
                pass
    assets["slang_dict"] = slang_dict

    # 5. Muat Lexicon (Hibrida)
    final_lexicon = {}
    lexicon_sources = {
        'pos_tsv': {'url': 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv', 'local': 'assets/positive.tsv', 'params': {'sep': '\t', 'header': None, 'names': ['word']}, 'weight': 1},
        'pos_csv': {'url': 'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv', 'local': 'assets/lexicon_positive.csv', 'params': {'header': None, 'names': ['word', 'score']}, 'weight': 1},
        'neg_tsv': {'url': 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv', 'local': 'assets/negative.tsv', 'params': {'sep': '\t', 'header': None, 'names': ['word']}, 'weight': -1},
        'neg_csv': {'url': 'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv', 'local': 'assets/lexicon_negative.csv', 'params': {'header': None, 'names': ['word', 'score']}, 'weight': -1},
    }
    for source_name, source in lexicon_sources.items():
        try:
            df_lex = pd.read_csv(source['url'], **source['params'])
        except Exception:
            try:
                df_lex = pd.read_csv(source['local'], **source['params'])
            except Exception:
                continue

        if not df_lex.empty and 'word' in df_lex.columns:
            if 'score' in df_lex.columns:
                for _, row in df_lex.iterrows():
                    word = str(row['word']).strip().lower()
                    try:
                        score = float(row['score'])
                    except (ValueError, TypeError):
                        score = source['weight']
                    final_lexicon[word] = final_lexicon.get(word, 0) + score

            else:
                for word in df_lex['word'].dropna().astype(str).str.lower():
                    final_lexicon[word.strip()] = final_lexicon.get(word.strip(), 0) + source['weight']

    assets["lexicon"] = final_lexicon

    # 6. Muat Skor Model (opsional)
    try:
        with open('model_scores.json', 'r') as f:
            assets["model_scores"] = json.load(f)
    except FileNotFoundError:
        pass
    
    return assets

# --- Fungsi Pemuatan Model Deep Learning dengan Caching ---
@st.cache_resource
def load_all_models(model_dir='models'):
    """Memuat semua model .h5 dari direktori."""
    models = {}
    if not os.path.exists(model_dir):
        st.error(f"Direktori model '{model_dir}' tidak ditemukan.")
        return models
    for root, _, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith('.h5') and filename.startswith("best_model"):
                model_name = os.path.relpath(os.path.join(root, filename), model_dir).replace(".h5", "").replace("best_model_", "").replace("/", " - ")
                model_path = os.path.join(root, filename)
                try:
                    models[model_name] = tf.keras.models.load_model(model_path, compile=False)
                except Exception as e:
                    st.warning(f"Gagal memuat model {model_name}: {e}")
    return models

# --- Eksekusi Pemuatan Aset dan Model dengan Feedback UI ---
with st.spinner("Memuat aset (tokenizer, stopwords, lexicon) dan model..."):
    loaded_assets = load_all_assets()
    all_models = load_all_models()

# === Variabel dan Konfigurasi ===
MAX_SEQUENCE_LEN = 30
id_to_label = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
stem_cache = {}

# === Fungsi Preprocessing Terpusat ===
def preprocess_text(text):
    """
    Pipeline preprocessing teks yang WAJIB SAMA dengan saat pelatihan model.
    Urutan: Cleaning -> Normalisasi -> Tokenisasi -> Slang -> Stopword -> Stemming.
    """
    stemmer = loaded_assets["stemmer"]
    stopwords = loaded_assets["stopwords"]
    slang_dict = loaded_assets["slang_dict"]
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'rt[\s]+', '', text)
    text = re.sub(r'[^a-z\s-]', '', text)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    tokens = nltk.tokenize.word_tokenize(text)
    
    processed_tokens = [slang_dict.get(token, token) for token in tokens if slang_dict.get(token, token) not in stopwords]
    
    if not stemmer:
        return ' '.join(processed_tokens)
    
    stemmed_tokens = []
    for word in processed_tokens:
        if word in stem_cache:
            stemmed_word = stem_cache[word]
        else:
            stemmed_word = stemmer.stem(word)
            stem_cache[word] = stemmed_word
        stemmed_tokens.append(stemmed_word)
    
    return ' '.join(stemmed_tokens)

# --- Fungsi Prediksi Deep Learning ---
def predict_sentiment_all_models(processed_text):
    if not loaded_assets["tokenizer"] or not all_models:
        return []
    
    tokenizer = loaded_assets["tokenizer"]
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_SEQUENCE_LEN, padding='post', truncating='post')

    predictions = []
    for model_name, model_obj in all_models.items():
        try:
            prediction = model_obj.predict(padded_sequence, verbose=0)
            predicted_class_id = np.argmax(prediction, axis=1)[0]
            predictions.append({
                'model_name': model_name,
                'predicted_sentiment': id_to_label.get(predicted_class_id, "Tidak Dikenal"),
                'confidence': np.max(prediction)
            })
        except Exception:
            predictions.append({'model_name': model_name, 'predicted_sentiment': "Error", 'confidence': 0})
    
    return sorted(predictions, key=lambda x: x['confidence'], reverse=True)

# --- Fungsi Analisis Lexicon ---
def sentiment_analysis_lexicon(text):
    lexicon_combined = loaded_assets["lexicon"]
    score = 0
    words_found = {"positif": [], "negatif": []}
    
    processed_text = preprocess_text(text)
    tokens = processed_text.split()
    
    for token in tokens:
        word_score = lexicon_combined.get(token, 0)
        if word_score > 0:
            words_found["positif"].append(token)
        elif word_score < 0:
            words_found["negatif"].append(token)
        score += word_score

    if score > 0: polarity = 'Positif'
    elif score < 0: polarity = 'Negatif'
    else: polarity = 'Netral'
    
    return score, polarity, words_found, processed_text

# --- Fungsi Highlight Kata (Lexicon) ---
def highlight_sentiment_words(original_text, processed_text):
    lexicon_combined = loaded_assets["lexicon"]
    highlighted_parts = []
    original_tokens = original_text.split()
    
    # Gunakan cache stemming untuk mengoptimalkan
    processed_sentiments = {word: lexicon_combined.get(word, 0) for word in processed_text.split()}
    
    for ori_token in original_tokens:
        # Hapus tanda baca dari token asli untuk perbandingan yang lebih baik
        cleaned_ori_token = re.sub(f'[{re.escape(string.punctuation)}]', '', ori_token).lower()
        
        # Stemming dan normalisasi token asli
        stemmed_ori_token = preprocess_text(cleaned_ori_token)
        
        # Cek apakah versi stemmed ada di lexicon
        score = lexicon_combined.get(stemmed_ori_token, 0)
        
        if score > 0:
            highlighted_parts.append(f'<span style="background-color: #d4edda; color: #155724; padding: 2px 5px; border-radius: 5px;">{ori_token}</span>')
        elif score < 0:
            highlighted_parts.append(f'<span style="background-color: #f8d7da; color: #721c24; padding: 2px 5px; border-radius: 5px;">{ori_token}</span>')
        else:
            highlighted_parts.append(ori_token)
            
    return ' '.join(highlighted_parts)

# === TAMPILAN UTAMA APLIKASI GUI ===
st.title("📊 Analisis Sentimen Teks Bahasa Indonesia")
st.markdown("Aplikasi ini menggunakan metode **Deep Learning** dan **Lexicon-Based** untuk menganalisis sentimen dari sebuah teks ulasan (Positif, Negatif, atau Netral).")

st.markdown("---")

# Area Input Pengguna
user_input = st.text_area("📝 Masukkan teks ulasan atau komentar Anda di sini:", 
                          height=150, 
                          placeholder="Contoh: Aplikasi ini luar biasa, fiturnya sangat membantu!")

if st.button("Analisis Sentimen", type="primary"):
    if not user_input.strip():
        st.warning("Mohon masukkan teks terlebih dahulu untuk dianalisis.")
    else:
        if not all_models:
            st.error("Tidak ada model Deep Learning yang berhasil dimuat. Hanya analisis Lexicon yang akan tersedia.")
        
        with st.spinner("Sedang menganalisis..."):
            # Lakukan analisis lexicon & preprocessing sekali saja
            lex_score, lex_polarity, words_found, cleaned_text = sentiment_analysis_lexicon(user_input)
            
            # Gunakan teks yang sudah diproses untuk prediksi model
            model_predictions = predict_sentiment_all_models(cleaned_text)
            
            # Buat highlight
            highlighted_text = highlight_sentiment_words(user_input, cleaned_text)
        
        st.markdown("---")
        st.subheader("🎉 Hasil Analisis")

        col1, col2 = st.columns(2)
        with col1:
            st.info("**Teks Asli Anda:**")
            st.write(f"_{user_input}_")
        with col2:
            st.info("**Teks Setelah Diproses AI:**")
            st.write(f"_{cleaned_text}_")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tampilkan hasil Deep Learning dan Lexicon dalam kolom
        col_dl, col_lex = st.columns(2)
        with col_dl:
            st.markdown("##### 🤖 **Hasil Deep Learning**")
            if model_predictions:
                best_pred = model_predictions[0]
                pred = best_pred['predicted_sentiment']
                confidence_str = f"{best_pred['confidence']:.2%}"

                if pred == "Positif": st.success(f"**{pred}**")
                elif pred == "Negatif": st.error(f"**{pred}**")
                else: st.info(f"**{pred}**")
                st.caption(f"Model: *{best_pred['model_name']}*, Keyakinan: {confidence_str}")
            else:
                st.warning("Tidak dapat melakukan prediksi karena tidak ada model yang dimuat.")
        
        with col_lex:
            st.markdown("##### 📚 **Hasil Lexicon-Based**")
            if lex_polarity == "Positif": st.success(f"**{lex_polarity}**")
            elif lex_polarity == "Negatif": st.error(f"**{lex_polarity}**")
            else: st.info(f"**{lex_polarity}**")
            st.caption(f"Total Skor: {lex_score}")

        st.markdown("---")
        st.markdown("#### ✨ Highlight Kata Berdasarkan Lexicon")
        st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)

        # --- Expander untuk detail ---
        with st.expander("🔍 Lihat Detail Analisis"):
            # Detail Prediksi Deep Learning
            if model_predictions:
                st.markdown("**Detail Prediksi Semua Model Deep Learning:**")
                sentiment_emoji_map = {'Positif': '😊', 'Negatif': '😠', 'Netral': '😐', 'Tidak Dikenal': '❓', 'Error': '❌'}
                
                cols = st.columns(len(model_predictions))
                for i, res in enumerate(model_predictions):
                    with cols[i]:
                        st.metric(
                            label=f"📦 Model: {res['model_name']}", 
                            value=f"{sentiment_emoji_map.get(res['predicted_sentiment'], '❓')} {res['predicted_sentiment']}",
                            delta=f"Kepercayaan: {res['confidence']:.2%}",
                            delta_color="off"
                        )
            
            st.markdown("<br>", unsafe_allow_html=True)

            # Detail Lexicon
            st.markdown("**Detail Kata Ditemukan (Lexicon):**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Kata Positif Ditemukan", len(words_found['positif']))
                st.write(', '.join(words_found['positif']) or "_Tidak ada_")
            with c2:
                st.metric("Kata Negatif Ditemukan", len(words_found['negatif']))
                st.write(', '.join(words_found['negatif']) or "_Tidak ada_")

# --- Grafik Perbandingan (jika ada data) ---
model_scores = loaded_assets.get('model_scores', {})
if model_scores:
    st.markdown("---")
    st.subheader("📈 Perbandingan Kinerja Model")
    df_scores = pd.DataFrame.from_dict(model_scores, orient='index')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_scores.index,
        y=df_scores['accuracy'] * 100,
        name='Accuracy',
        marker_color='royalblue'
    ))
    fig.add_trace(go.Bar(
        x=df_scores.index,
        y=df_scores['f1'] * 100,
        name='F1-Score',
        marker_color='lightsalmon'
    ))
    
    fig.update_layout(
        title_text="Perbandingan Accuracy dan F1-Score Antar Model (Berdasarkan Data Uji)",
        barmode='group',
        xaxis_tickangle=-45,
        yaxis_title="Persentase (%)"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("Dibuat dengan Streamlit dan TensorFlow")