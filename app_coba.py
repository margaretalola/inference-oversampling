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

st.set_page_config(page_title="Analisis Sentimen Indonesia", layout="wide")

# --- Download Resource NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.info("Mendownload resource NLTK 'punkt'...")
    nltk.download('punkt')

def load_model_assets():
    """
    Memuat aset dengan pendekatan HIBRIDA yang disempurnakan:
    1. Mencoba dari URL online terlebih dahulu.
    2. Jika gagal, beralih ke file cadangan LOKAL.
    3. Cerdas dalam membaca berbagai format file (CSV/TSV, dengan/tanpa header).
    """
    assets = {
        "tokenizer": None, "stopwords": set(), "slang_dict": {},
        "lexicon": {}, "model_scores": {}, "stemmer": None
    }
    
    # --- 1. Muat Aset Inti (Tokenizer & Stemmer tetap lokal karena kritis) ---
    try:
        factory = StemmerFactory()
        assets["stemmer"] = factory.create_stemmer()
        st.success("Stemmer berhasil diinisialisasi.")
    except Exception as e:
        st.warning(f"Gagal inisialisasi Stemmer: {e}")

    tokenizer_path = 'models/tokenizer.pkl'
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            assets["tokenizer"] = pickle.load(handle)
        st.success(f"Tokenizer berhasil dimuat dari '{tokenizer_path}'.")
    else:
        st.error(f"Kritis: File '{tokenizer_path}' tidak ditemukan.")

    # --- 2. Muat Stopwords dari LOKAL (lebih cepat dan andal) ---
    try:
        with open('assets/indonesian-stopwords-complete.txt', 'r', encoding='utf-8') as f:
            assets["stopwords"] = set(f.read().splitlines())
        st.success("Stopwords berhasil dimuat dari file lokal.")
    except Exception as e:
        st.error(f"Gagal memuat stopwords dari 'assets/indonesian-stopwords-complete.txt': {e}")
    
    # --- 3. Muat Kamus Slang (Hibrida) ---
    slang_dict = {}
    st.info("Memuat kamus slang...")

    # Sumber Slang 1 (CSV dengan header)
    try:
        df_slang_1 = pd.read_csv('https://github.com/adeariniputri/text-preprocesing/raw/master/slang.csv')
        st.success("Slang 1 (CSV) berhasil dimuat dari URL.")
    except Exception as e:
        st.warning(f"Gagal memuat Slang 1 dari URL. Mencoba dari file lokal 'assets/slang.csv'...")
        try:
            df_slang_1 = pd.read_csv('assets/slang.csv')
            st.success("Slang 1 (CSV) berhasil dimuat dari LOKAL.")
        except Exception as e_local:
            st.error(f"GAGAL memuat Slang 1: {e_local}")
            df_slang_1 = pd.DataFrame()
    
    if not df_slang_1.empty:
        for _, row in df_slang_1.iterrows():
            slang_dict[str(row.get('slang', '')).strip()] = str(row.get('formal', '')).strip()

    # Sumber Slang 2 (TXT tanpa header, pemisah ':')
    try:
        df_slang_2 = pd.read_csv('https://github.com/louisowen6/NLP_bahasa_resources/raw/master/combined_slang_words.txt', sep=":", header=None, names=["slang", "formal"])
        st.success("Slang 2 (TXT) berhasil dimuat dari URL.")
    except Exception as e:
        st.warning(f"Gagal memuat Slang 2 dari URL. Mencoba dari file lokal 'assets/combined_slang_words.txt'...")
        try:
            df_slang_2 = pd.read_csv('assets/combined_slang_words.txt', sep=":", header=None, names=["slang", "formal"])
            st.success("Slang 2 (TXT) berhasil dimuat dari LOKAL.")
        except Exception as e_local:
            st.error(f"GAGAL memuat Slang 2: {e_local}")
            df_slang_2 = pd.DataFrame()
    
    if not df_slang_2.empty:
        for _, row in df_slang_2.iterrows():
            slang_dict[str(row.get('slang', '')).strip()] = str(row.get('formal', '')).strip()
            
    assets["slang_dict"] = slang_dict
    st.info(f"Total slang words dimuat: {len(slang_dict)}")

    # --- 4. Muat Lexicon (Hibrida) ---
    final_lexicon = {}
    st.info("Memuat lexicon sentimen...")

    lexicon_sources = {
        'pos_tsv': {'url': 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv', 'local': 'assets/positive.tsv', 'params': {'sep': '\t', 'header': None, 'names': ['word']}, 'type': 'implicit_score', 'weight': 1},
        'pos_csv': {'url': 'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv', 'local': 'assets/lexicon_positive.csv', 'params': {'header': None, 'names': ['word', 'score']}, 'type': 'explicit_score', 'weight': 1},
        'neg_tsv': {'url': 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv', 'local': 'assets/negative.tsv', 'params': {'sep': '\t', 'header': None, 'names': ['word']}, 'type': 'implicit_score', 'weight': -1},
        'neg_csv': {'url': 'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv', 'local': 'assets/lexicon_negative.csv', 'params': {'header': None, 'names': ['word', 'score']}, 'type': 'explicit_score', 'weight': -1},
    }

    for name, source in lexicon_sources.items():
        df_lex = pd.DataFrame()
        try:
            df_lex = pd.read_csv(source['url'], **source['params'])
            st.success(f"Lexicon '{name}' berhasil dimuat dari URL.")
        except Exception as e:
            st.warning(f"Gagal memuat '{name}' dari URL. Mencoba dari file lokal...")
            try:
                df_lex = pd.read_csv(source['local'], **source['params'])
                st.success(f"Lexicon '{name}' berhasil dimuat dari LOKAL.")
            except Exception as e_local:
                st.error(f"GAGAL memuat '{name}': {e_local}")
        
        if not df_lex.empty:
            if source['type'] == 'implicit_score':
                if 'word' in df_lex.columns:
                    for word in df_lex['word'].dropna().astype(str).str.lower():
                        final_lexicon[word.strip()] = final_lexicon.get(word.strip(), 0) + source['weight']
                else:
                    st.error(f"Kolom 'word' tidak ditemukan di lexicon '{name}'. Periksa format file dan 'names' parameter.")
            
            elif source['type'] == 'explicit_score':
                if 'word' in df_lex.columns and 'score' in df_lex.columns:
                    for index, row in df_lex.iterrows():
                        word = str(row['word']).strip().lower()
                        try:
                            score = float(row['score'])
                        except ValueError:
                            st.warning(f"Skipping row in '{name}' due to non-numeric score: '{row['score']}' for word '{word}'")
                            continue
                        final_lexicon[word] = final_lexicon.get(word, 0) + (score * source['weight'])
                else:
                    st.error(f"Kolom 'word' atau 'score' tidak ditemukan di lexicon '{name}'. Periksa format file dan 'names' parameter.")
        else:
            st.warning(f"DataFrame for '{name}' is empty. No words loaded from this source.")

    # Hapus stopword dari final lexicon
    for stopword in assets["stopwords"]:
        if stopword in final_lexicon:
            del final_lexicon[stopword]

    assets["lexicon"] = final_lexicon
    st.info(f"Proses pemuatan Lexicon selesai. Total kata unik yang dimuat: **{len(assets['lexicon'])}**")
            
    # --- 5. Muat Skor Model (opsional) ---
    try:
        with open('model_scores.json', 'r') as f:
            assets["model_scores"] = json.load(f)
    except FileNotFoundError:
        st.info("File 'model_scores.json' tidak ditemukan. Grafik perbandingan tidak akan ditampilkan.")
        
    return assets


assets = load_model_assets()
tokenizer = assets["tokenizer"]
stopwords = assets["stopwords"]
slang_dict = assets["slang_dict"]
lexicon_combined = assets["lexicon"]
model_scores = assets["model_scores"]
stemmer = assets["stemmer"]
stem_cache = {} # Inisialisasi cache stemming

# --- Pipeline Preprocessing yang Disatukan dan Diperbaiki ---
def normalize_elongated(text):
    text = re.sub(r'(\w)-(\1-)+', r'\1', text)
    return re.sub(r'(\w)\1{2,}', r'\1\1', text)

# Fungsi preprocessing yang SAMA PERSIS dengan yang di terminal
# Ganti nama 'preprocess_text_unified' menjadi 'preprocess_text'
def preprocess_text(text):
    """
    Satu fungsi untuk semua kebutuhan preprocessing.
    Urutan: Cleaning -> Normalisasi -> Tokenisasi -> Stopword/Slang -> Stemming.
    Fungsi ini WAJIB SAMA dengan yang digunakan saat membuat tokenizer.
    """
    # 1. Cleaning dan Case Folding
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text) # Hapus URL, mention, hashtag
    text = re.sub(r'rt[\s]+', '', text) # Hapus RT
    text = re.sub(r'[^a-z\s-]', '', text) # Hapus karakter non-alfanumerik kecuali spasi dan strip
    
    # 2. Normalisasi kata yang dipanjangkan (e.g., baguuus -> baguus)
    text = normalize_elongated(text)
    
    # 3. Tokenisasi
    tokens = word_tokenize(text)
    
    # 4. Normalisasi Slang dan Penghapusan Stopword
    processed_tokens = []
    for token in tokens:
        # Periksa slang dulu
        normalized_token = slang_dict.get(token, token)
        # Hapus stopword setelah normalisasi
        if normalized_token and normalized_token not in stopwords:
            processed_tokens.append(normalized_token)
            
    # 5. Stemming (jika stemmer tersedia)
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


# --- Fungsi Analisis & Prediksi ---
def sentiment_analysis_lexicon(text):
    """Menganalisis sentimen berdasarkan lexicon setelah preprocessing."""
    score = 0
    words_found = {"positif": [], "negatif": []}
    
    # Menggunakan fungsi preprocessing yang sudah disatukan (sekarang bernama preprocess_text)
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

def highlight_sentiment_words(original_text, processed_text):
    """Menyorot kata di teks asli berdasarkan token yang sudah diproses."""
    highlighted_parts = []
    original_tokens = original_text.split()
    processed_tokens = processed_text.split()
    
    # Karena preprocessing bisa mengubah jumlah token, metode ini bersifat aproksimasi
    # Ini akan menyorot kata asli jika versi stem-nya ada di lexicon
    processed_sentiments = {token: lexicon_combined.get(token, 0) for token in processed_tokens}

    for ori_token in original_tokens:
        # Buat versi proses dari token asli untuk perbandingan
        cleaned_ori_token = preprocess_text(ori_token) # Pastikan ini juga memanggil fungsi yang benar
        
        score = processed_sentiments.get(cleaned_ori_token, 0)

        if score > 0:
            highlighted_parts.append(f'<span style="background-color: #d4edda; color: #155724; padding: 2px 5px; border-radius: 5px;">{ori_token}</span>')
        elif score < 0:
            highlighted_parts.append(f'<span style="background-color: #f8d7da; color: #721c24; padding: 2px 5px; border-radius: 5px;">{ori_token}</span>')
        else:
            highlighted_parts.append(ori_token)
            
    return ' '.join(highlighted_parts)

def load_all_models_cached(model_dir='models'):
    """Memuat semua model .h5 dari direktori dan menyimpannya di cache."""
    loaded_models = {}
    if not os.path.isdir(model_dir):
        st.error(f"Folder '{model_dir}' tidak ditemukan.")
        return loaded_models
    for root, _, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith('.h5') and filename.startswith("best_model"):
                model_path = os.path.join(root, filename)
                model_name = os.path.splitext(filename)[0]
                try:
                    loaded_models[model_name] = tf.keras.models.load_model(model_path)
                except Exception as e:
                    st.warning(f"Gagal memuat model {filename}: {e}")
    return loaded_models

loaded_models = load_all_models_cached()

# --- Fungsi Prediksi Deep Learning ---
MAX_SEQUENCE_LEN_DL = 30 
label_map_dl = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

def predict_sentiment_all_models(processed_text):
    """Melakukan prediksi pada semua model yang dimuat."""
    if not tokenizer or not loaded_models:
        st.error("Tokenizer atau Model tidak dimuat.")
        return [], ""
        
    # Teks sudah diproses sebelumnya, tidak perlu diulang
    seq = tokenizer.texts_to_sequences([processed_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_SEQUENCE_LEN_DL)
    
    # Cek OOV
    tokens = processed_text.split()
    oov = [t for t in tokens if t not in tokenizer.word_index]
    
    predictions = []
    for name, model in loaded_models.items():
        try:
            pred = model.predict(padded, verbose=0)
            label_idx = int(np.argmax(pred))
            confidence = float(np.max(pred))
            predictions.append({
                "Model": name,
                "Prediksi": label_map_dl.get(label_idx, "Unknown"),
                "Keyakinan": confidence * 100
            })
        except Exception:
            predictions.append({"Model": name, "Prediksi": "Error", "Keyakinan": 0})
            
    return sorted(predictions, key=lambda x: x['Keyakinan'], reverse=True), oov

# --- UI STREAMLIT ---
st.title("🔍 Analisis Sentimen Teks Bahasa Indonesia")
st.markdown("Masukkan teks ulasan, komentar, atau cuitan untuk dianalisis sentimennya menggunakan metode berbasis Lexicon dan Deep Learning.")

input_text = st.text_area("📝 Masukkan teks di sini:", "aplikasi ini sangat bagus, saya suka sekali fitur-fiturnya!", height=100)

if st.button("Analisis Sekarang!"):
    if not input_text.strip():
        st.warning("Teks tidak boleh kosong!")
    else:
        with st.spinner("Menganalisis..."):
            # Lakukan analisis lexicon & preprocessing sekali saja
            lex_score, lex_polarity, words_found, cleaned_display = sentiment_analysis_lexicon(input_text)
            
            # Gunakan teks yang sudah diproses untuk prediksi model
            model_predictions, oov_words = predict_sentiment_all_models(cleaned_display)
            
            # Buat highlight
            highlighted_text = highlight_sentiment_words(input_text, cleaned_display)

        st.subheader("🎉 Hasil Analisis")
        st.markdown(f"**Teks Asli:** *{input_text}*")
        st.markdown(f"**Setelah Preprocessing:** `{cleaned_display}`")

        # Tampilkan peringatan OOV jika ada
        if oov_words:
            st.warning(f"**Kata Tidak Dikenal (OOV):** `{', '.join(oov_words)}`. Kata-kata ini tidak ada dalam vocabulary model dan mungkin mempengaruhi akurasi.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🤖 **Deep Learning**")
            if model_predictions:
                best_pred = model_predictions[0]
                pred = best_pred['Prediksi']
                confidence_str = f"{best_pred['Keyakinan']:.2f}%"

                if pred == "Positif": st.success(f"**{pred}**")
                elif pred == "Negatif": st.error(f"**{pred}**")
                else: st.info(f"**{pred}**")
                st.caption(f"Model: *{best_pred['Model']}*, Keyakinan: {confidence_str}")
        with col2:
            st.markdown("##### 📚 **Lexicon-Based**")
            if lex_polarity == "Positif": st.success(f"**{lex_polarity}**")
            elif lex_polarity == "Negatif": st.error(f"**{lex_polarity}**")
            else: st.info(f"**{lex_polarity}**")
            st.caption(f"Total Skor: {lex_score}")

        st.markdown("---")
        st.markdown("#### ✨ Highlight Kata Berdasarkan Lexicon")
        st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)

        # --- Expander untuk detail ---
        with st.expander("🔍 Lihat Detail Analisis"):
            st.markdown("**Detail Kata Ditemukan (Lexicon):**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Kata Positif Ditemukan", len(words_found['positif']))
                st.write(', '.join(words_found['positif']) or "_Tidak ada_")
            with c2:
                st.metric("Kata Negatif Ditemukan", len(words_found['negatif']))
                st.write(', '.join(words_found['negatif']) or "_Tidak ada_")

            if model_predictions:
                st.markdown("**Detail Prediksi Semua Model Deep Learning:**")
                df_preds = pd.DataFrame(model_predictions)
                # Format kolom keyakinan
                df_preds['Keyakinan'] = df_preds['Keyakinan'].map('{:.2f}%'.format)
                st.dataframe(df_preds, use_container_width=True)

# --- Grafik Perbandingan (jika ada data) ---
if model_scores:
    st.subheader("📈 Perbandingan Kinerja Model (Berdasarkan Data Uji)")
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
        title_text="Perbandingan Accuracy dan F1-Score Antar Model",
        barmode='group',
        xaxis_tickangle=-45,
        yaxis_title="Persentase (%)"
    )
    st.plotly_chart(fig, use_container_width=True)
