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
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import defaultdict

# --- 1. Konfigurasi Halaman & Download Resource ---
st.set_page_config(
    page_title="Analisis Sentimen Profesional",
    page_icon="✨",
    layout="wide"
)

# Download resource NLTK 'punkt' jika belum ada, dengan caching
@st.cache_resource
def download_nltk_resource():
    """Mendownload resource NLTK 'punkt' yang diperlukan untuk tokenisasi."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

download_nltk_resource()

# --- 2. Fungsi Pemuatan Aset & Model (Menggunakan Cache) ---

class CustomSastrawiStemmer:
    """
    Sebuah wrapper untuk Sastrawi Stemmer yang memungkinkan kita menambahkan
    kamus pengecualian untuk menangani kata-kata yang salah di-stem.
    """
    def __init__(self, custom_dictionary):
        factory = StemmerFactory()
        self.base_stemmer = factory.create_stemmer()
        self.custom_dictionary = custom_dictionary

    def stem(self, term):
        """
        Melakukan stemming. Jika kata ada di kamus kustom, kembalikan nilai dari kamus.
        Jika tidak, gunakan stemmer Sastrawi standar.
        """
        if term in self.custom_dictionary:
            return self.custom_dictionary[term]
        return self.base_stemmer.stem(term)

@st.cache_resource
def load_all_assets():
    """
    Memuat semua aset penting (tokenizer, stopwords, lexicon, stemmer) dengan caching.
    Fungsi ini dijalankan sekali saat aplikasi pertama kali dimuat.
    """
    assets = {
        "tokenizer": None, "stopwords": set(), "slang_dict": {},
        "lexicon": {}, "lexicon_transparency": {}, "model_scores": {}, "stemmer": None
    }
    
    # Muat Stemmer Sastrawi
    try:
        custom_dictionary = {'pengalaman': 'pengalaman', 'penggunaan': 'guna', 'makanan': 'makan', 'pelayanan': 'pelayanan'}
        assets["stemmer"] = CustomSastrawiStemmer(custom_dictionary)
    except Exception as e:
        st.error(f"KRITIS: Gagal memuat Stemmer. Error: {e}")

    # Muat Stopwords
    try:
        with open('assets/indonesian-stopwords-complete.txt', 'r', encoding='utf-8') as f:
            assets["stopwords"] = set(f.read().splitlines())
    except Exception as e:
        st.warning(f"Gagal memuat stopwords: {e}")

    # Muat Kamus Slang
    try:
        df_slang = pd.read_csv('https://raw.githubusercontent.com/adeariniputri/text-preprocesing/master/slang.csv')
    except Exception:
        try:
            df_slang = pd.read_csv('assets/slang.csv')
        except Exception: df_slang = pd.DataFrame()
    slang_dict = {str(row.get('slang', '')).strip(): str(row.get('formal', '')).strip() for _, row in df_slang.iterrows()}
    manual_slang_additions = {
        "mantep": "mantap", "bgt": "banget", "gak": "tidak", "ga": "tidak", "ngga": "tidak",
        "kalo": "kalau", "krn": "karena", "yg": "yang", "sm": "sama", "jg": "juga", "gue": "saya", "gw": "saya"
    }
    slang_dict.update(manual_slang_additions)
    assets["slang_dict"] = slang_dict

    # PERUBAHAN: Mengintegrasikan pemuatan leksikon dengan transparansi di sini.
    final_lexicon = defaultdict(float)
    lexicon_transparency = defaultdict(lambda: defaultdict(float))
    
    lexicon_sources = {
        'InSet_Pos': {'url': 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv', 'local': 'assets/positive.tsv', 'params': {'sep': '\t', 'header': None, 'names': ['word']}, 'weight': 1},
        'InSet_Neg': {'url': 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv', 'local': 'assets/negative.tsv', 'params': {'sep': '\t', 'header': None, 'names': ['word']}, 'weight': -1},
        'Custom_Pos': {'url': 'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv', 'local': 'assets/lexicon_positive.csv', 'params': {'header': None, 'names': ['word', 'score']}, 'weight': 1},
        'Custom_Neg': {'url': 'https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv', 'local': 'assets/lexicon_negative.csv', 'params': {'header': None, 'names': ['word', 'score']}, 'weight': -1},
    }

    for source_name, source in lexicon_sources.items():
        try:
            df_lex = pd.read_csv(source['url'], **source['params'])
        except Exception:
            try:
                df_lex = pd.read_csv(source['local'], **source['params'])
            except Exception: continue

        for _, row in df_lex.iterrows():
            word = str(row['word']).strip().lower()
            if not word: continue
            
            score = float(row.get('score', source['weight']))
            final_lexicon[word] += score
            lexicon_transparency[word][source_name] += score
            
    assets["lexicon"] = dict(final_lexicon)
    assets["lexicon_transparency"] = dict(lexicon_transparency)


    # Muat Tokenizer
    try:
        with open('models/tokenizer.pkl', 'rb') as handle:
            assets["tokenizer"] = pickle.load(handle)
    except FileNotFoundError:
        st.error("KRITIS: File 'models/tokenizer.pkl' tidak ditemukan. Prediksi tidak dapat dilakukan.")
        st.stop()

    # Muat Skor Model
    try:
        with open('model_scores.json', 'r') as f:
            assets["model_scores"] = json.load(f)
    except FileNotFoundError: pass
    
    return assets

@st.cache_resource
def load_all_models(model_dir='models'):
    """Memuat semua model .h5 dari direktori target dengan caching."""
    models = {}
    if not os.path.exists(model_dir):
        st.error(f"Direktori model '{model_dir}' tidak ditemukan.")
        return models
    for filename in os.listdir(model_dir):
        if filename.endswith('.h5'):
            model_path = os.path.join(model_dir, filename)
            model_name = filename.replace('.h5', '').replace('best_model_', '').replace('_', ' ').title()
            try:
                models[model_name] = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e:
                st.warning(f"Gagal memuat model '{model_name}': {e}")
    return models

# --- 3. Eksekusi Pemuatan dengan Tampilan UI ---
with st.spinner("Mempersiapkan semua aset dan model AI... Ini hanya dilakukan sekali saat aplikasi dibuka."):
    loaded_assets = load_all_assets()
    all_models = load_all_models()

# --- 4. Variabel Global & Fungsi-Fungsi Inti ---
MAX_SEQUENCE_LEN = 30
ID_TO_LABEL = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
SENTIMENT_EMOJI_MAP = {'Positif': '😊', 'Negatif': '😠', 'Netral': '😐', 'Tidak Dikenal': '❓'}

def preprocess_text(text, return_steps=False):
    """Pipeline preprocessing teks yang WAJIB SAMA dengan saat pelatihan model."""
    if not isinstance(text, str): return "" if not return_steps else {}
    steps = {'original': text}
    processed = text.lower()
    steps['case_folding'] = processed
    processed = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', processed)
    processed = re.sub(r'\d+', '', processed)
    processed = re.sub(f'[{re.escape(string.punctuation)}]+', ' ', processed)
    processed = re.sub(r'(\w)\1{2,}', r'\1', processed)
    processed = re.sub(r'\s+', ' ', processed).strip()
    steps['cleaning'] = processed
    tokens = word_tokenize(processed)
    normalized_tokens = [loaded_assets["slang_dict"].get(token, token) for token in tokens]
    steps['normalization'] = ' '.join(normalized_tokens)
    stopped_tokens = [token for token in normalized_tokens if token not in loaded_assets["stopwords"]]
    steps['stopword_removal'] = ' '.join(stopped_tokens)
    stemmed_tokens = [loaded_assets["stemmer"].stem(word) for word in stopped_tokens]
    final_text = ' '.join(stemmed_tokens)
    steps['stemming'] = final_text
    return steps if return_steps else final_text

def predict_sentiment_all_models(processed_text):
    """Melakukan prediksi sentimen menggunakan semua model yang berhasil dimuat."""
    if not loaded_assets["tokenizer"] or not all_models: return []
    tokenizer = loaded_assets["tokenizer"]
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_SEQUENCE_LEN, padding='post', truncating='post')
    predictions = []
    for model_name, model_obj in all_models.items():
        try:
            prediction = model_obj.predict(padded_sequence, verbose=0)
            pred_id = np.argmax(prediction, axis=1)[0]
            predictions.append({
                'model_name': model_name, 'sentiment': ID_TO_LABEL.get(pred_id, "Tidak Dikenal"), 'confidence': np.max(prediction)
            })
        except Exception:
            predictions.append({'model_name': model_name, 'sentiment': "Error", 'confidence': 0})
    return sorted(predictions, key=lambda x: x['confidence'], reverse=True)

# PERUBAHAN: Fungsi analisis leksikon yang diperbaiki untuk transparansi.
def analyze_sentiment_lexicon(processed_text):
    """
    Menganalisis sentimen berdasarkan skor dari kamus lexicon dan memberikan detail transparansi.
    Mengembalikan total skor, polaritas, dan detail kata yang ditemukan.
    """
    lexicon = loaded_assets["lexicon"]
    lexicon_transparency = loaded_assets["lexicon_transparency"]
    
    total_score = 0
    word_details = []

    for token in set(processed_text.split()): # Gunakan set untuk menghindari duplikasi kata
        if token in lexicon:
            word_score = lexicon[token]
            total_score += word_score
            word_details.append({
                "token": token,
                "score": word_score,
                "sources": lexicon_transparency.get(token, {})
            })

    if total_score > 0: polarity = 'Positif'
    elif total_score < 0: polarity = 'Negatif'
    else: polarity = 'Netral'
    
    return total_score, polarity, word_details

def highlight_sentiment_words(original_text):
    """Menyorot kata dalam teks asli berdasarkan sentimen dari lexicon."""
    highlighted_parts = []
    for token in original_text.split():
        cleaned_token = re.sub(f'[{re.escape(string.punctuation)}]', '', token).lower()
        if not cleaned_token:
            highlighted_parts.append(token)
            continue
        stemmed_token = loaded_assets["stemmer"].stem(cleaned_token)
        score = loaded_assets["lexicon"].get(cleaned_token, loaded_assets["lexicon"].get(stemmed_token, 0))
        if score > 0:
            highlighted_parts.append(f'<span style="background-color: #d4edda; color: #155724; padding: 2px 5px; border-radius: 5px; font-weight: bold;" title="Skor: {score:.2f}">{token}</span>')
        elif score < 0:
            highlighted_parts.append(f'<span style="background-color: #f8d7da; color: #721c24; padding: 2px 5px; border-radius: 5px; font-weight: bold;" title="Skor: {score:.2f}">{token}</span>')
        else:
            highlighted_parts.append(token)
    return ' '.join(highlighted_parts)

# --- 5. TAMPILAN APLIKASI (GUI) ---

with st.sidebar:
    st.title("Informasi Tambahan")
    st.divider()
    with st.expander("**Bagaimana Cara Kerjanya?**", expanded=False):
        st.write("""
        Aplikasi ini menggunakan dua pendekatan utama untuk menganalisis sentimen:
        1.  **Deep Learning:** Teks Anda diproses (cleaning, normalisasi, stemming) lalu diumpankan ke model AI (seperti LSTM & BiLSTM) untuk memprediksi sentimen.
        2.  **Lexicon-Based:** Setiap kata dalam teks dicocokkan dengan kamus sentimen untuk dihitung total skornya.
        Deep Learning umumnya lebih akurat karena mampu memahami konteks.
        """)
    st.subheader("Perbandingan Kinerja Umum Model")
    model_scores = loaded_assets.get('model_scores', {})
    if model_scores:
        st.markdown("Grafik ini menunjukkan perbandingan metrik evaluasi dari setiap model pada data uji terpisah.")
        df_scores = pd.DataFrame.from_dict(model_scores, orient='index').reset_index().rename(columns={'index': 'Model'})
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_scores['Model'],
            x=df_scores['accuracy'] * 100,
            name='Akurasi',
            orientation='h',
            marker_color='#1f77b4',
            text=df_scores['accuracy'].map('{:.1%}'.format),
            textposition='auto'
        ))
        fig.add_trace(go.Bar(
            y=df_scores['Model'],
            x=df_scores['f1'] * 100,
            name='F1-Score',
            orientation='h',
            marker_color='#ff7f0e',
            text=df_scores['f1'].map('{:.1%}'.format),
            textposition='auto'
        ))
        
        fig.update_layout(
            barmode='group',
            yaxis_title="Model",
            xaxis_title="Persentase (%)",
            legend_title_text="Metrik",
            margin=dict(l=40, r=40, t=30, b=20),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("File `model_scores.json` tidak ditemukan. Grafik perbandingan tidak dapat ditampilkan.")

    st.divider()
    st.write("Dibuat dengan Streamlit & TensorFlow.")


st.title("Analisis Sentimen")
st.markdown("Platform untuk menganalisis sentimen teks berbahasa Indonesia menggunakan metode **Deep Learning** dan **Lexicon-Based** secara real-time.")

main_container = st.container(border=True)

with main_container:
    user_input = st.text_area(
        "**Masukkan teks ulasan atau komentar Anda di sini:**",
        height=150,
        placeholder="Contoh: Aplikasi ini luar biasa, fiturnya sangat membantu dan mudah digunakan!"
    )
    
    # PERUBAHAN: Menghapus blok 'if text:' yang salah dan hanya menggunakan alur tombol.
    if st.button("Analisis Sekarang!", type="primary", use_container_width=True):
        if not user_input.strip() or len(user_input.split()) < 2:
            st.warning("Mohon masukkan minimal dua kata untuk hasil analisis yang lebih baik.")
        elif not all_models:
            st.error("Tidak ada model Deep Learning yang berhasil dimuat. Analisis tidak dapat dilanjutkan.")
        else:
            with st.spinner("sedang menganalisis teks..."):
                preprocessing_steps = preprocess_text(user_input, return_steps=True)
                final_processed_text = preprocessing_steps['stemming']
                
                model_predictions = predict_sentiment_all_models(final_processed_text)
                # PERUBAHAN: Memanggil fungsi lexicon yang sudah diperbaiki
                lex_score, lex_polarity, lex_details = analyze_sentiment_lexicon(final_processed_text)
                highlighted_text = highlight_sentiment_words(user_input)

            st.divider()
            st.header("Hasil Analisis Sentimen")

            best_prediction = model_predictions[0] if model_predictions else {'sentiment': 'Error', 'confidence': 0, 'model_name': 'N/A'}
            main_sentiment, main_confidence = best_prediction['sentiment'], best_prediction['confidence']
            main_emoji = SENTIMENT_EMOJI_MAP.get(main_sentiment, '❓')
            
            st.markdown(f"### Hasil Prediksi Utama: **{main_sentiment}** {main_emoji}")
            st.caption(f"Hasil dari model dengan keyakinan prediksi tertinggi pada teks ini: **{best_prediction['model_name']}** ({main_confidence:.2%}).")

            tab1, tab2, tab3, tab4 = st.tabs([
                "**Ringkasan & Sorotan**", 
                "**Detail Proses Teks**", 
                "**Detail Analisis Leksikon**", 
                "**Perbandingan Model AI**"
            ])

            with tab1:
                # ... (Tidak ada perubahan di tab ini)
                st.subheader("Perbandingan Hasil Metode")
                st.write("Perbandingan antara hasil prediksi model AI (Deep Learning) dengan metode berbasis kamus (Lexicon).")
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.markdown("<div style='text-align: center;'><strong>Deep Learning (Model Terbaik)</strong></div>", unsafe_allow_html=True)
                        if main_sentiment == "Positif": st.success(f"**{main_sentiment}** {main_emoji}", icon="✅")
                        elif main_sentiment == "Negatif": st.error(f"**{main_sentiment}** {main_emoji}", icon="❌")
                        else: st.info(f"**{main_sentiment}** {main_emoji}", icon="ℹ️")
                        st.metric(label="Tingkat Keyakinan", value=f"{main_confidence:.2%}")
                with col2:
                    with st.container(border=True):
                        st.markdown("<div style='text-align: center;'><strong>Lexicon-Based</strong></div>", unsafe_allow_html=True)
                        if lex_polarity == "Positif": st.success(f"**{lex_polarity}** ")
                        elif lex_polarity == "Negatif": st.error(f"**{lex_polarity}** ")
                        else: st.info(f"**{lex_polarity}**")
                        st.metric(label="Total Skor Leksikon", value=f"{lex_score:.2f}")
                st.divider()
                st.subheader("Sorotan Kata Berdasarkan Leksikon Sentimen")
                st.markdown("Arahkan kursor ke kata yang disorot untuk melihat skornya. Sorotan berdasarkan akar kata (stem) yang ditemukan di kamus.")
                st.markdown(f'<div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">{highlighted_text}</div>', unsafe_allow_html=True)


            with tab2:
                # ... (Tidak ada perubahan di tab ini)
                st.subheader("Langkah-langkah Preprocessing Teks")
                st.write("Berikut adalah tahapan pemrosesan yang dilalui teks Anda sebelum dianalisis oleh model AI.")
                with st.expander("Lihat Detail Langkah demi Langkah", expanded=True):
                    steps_html = f"""
                    <ol style="list-style-type: none; padding-left: 0;">
                        <li style="margin-bottom: 10px;"><b>1. Teks Asli:</b><br><i style="color: #555;">{preprocessing_steps['original']}</i></li>
                        <li style="margin-bottom: 10px;"><b>2. Cleaning & Case Folding:</b><br><i style="color: #555;">{preprocessing_steps['cleaning']}</i></li>
                        <li style="margin-bottom: 10px;"><b>3. Normalisasi (Kata Slang):</b><br><i style="color: #555;">{preprocessing_steps['normalization']}</i></li>
                        <li style="margin-bottom: 10px;"><b>4. Stopword Removal:</b><br><i style="color: #555;">{preprocessing_steps['stopword_removal']}</i></li>
                        <li style="margin-bottom: 10px; background-color: #e6ffed; padding: 8px; border-radius: 5px;">
                            <b>5. Hasil Akhir (Stemming):</b><br>
                            <strong style="color: #28a745;">{preprocessing_steps['stemming']}</strong>
                        </li>
                    </ol>
                    """
                    st.markdown(steps_html, unsafe_allow_html=True)
                st.caption("Teks hasil **Stemming** adalah input final yang digunakan untuk prediksi model Deep Learning.")


            with tab3:
                # PERUBAHAN: Menampilkan detail transparansi leksikon di sini.
                st.subheader("Detail Analisis Berbasis Leksikon & Transparansi")
                st.write("Rincian kata-kata bermuatan sentimen yang ditemukan dalam teks (setelah diproses) dan sumber skornya.")

                if not lex_details:
                    st.info("Tidak ada kata dari teks yang ditemukan dalam kamus leksikon.")
                else:
                    positif_words = [d for d in lex_details if d['score'] > 0]
                    negatif_words = [d for d in lex_details if d['score'] < 0]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Kata-kata Positif")
                        with st.container(border=True):
                            if not positif_words:
                                st.write("_Tidak ada kata positif yang ditemukan._")
                            for detail in sorted(positif_words, key=lambda x: x['score'], reverse=True):
                                sources_str = ", ".join([f"{src.replace('_', ' ')} ({score:+.1f})" for src, score in detail['sources'].items()])
                                st.success(f"**{detail['token']}** (Skor: {detail['score']:+.2f})")
                                st.caption(f"Sumber: {sources_str}")

                    with col2:
                        st.markdown("#### Kata-kata Negatif")
                        with st.container(border=True):
                            if not negatif_words:
                                st.write("_Tidak ada kata negatif yang ditemukan._")
                            for detail in sorted(negatif_words, key=lambda x: x['score']):
                                sources_str = ", ".join([f"{src.replace('_', ' ')} ({score:+.1f})" for src, score in detail['sources'].items()])
                                st.error(f"**{detail['token']}** (Skor: {detail['score']:+.2f})")
                                st.caption(f"Sumber: {sources_str}")

            with tab4:
                # ... (Tidak ada perubahan di tab ini)
                st.subheader("Perbandingan Prediksi Antar Model Deep Learning")
                st.write("Setiap model AI yang tersedia memberikan prediksinya masing-masing. Hasil dengan keyakinan tertinggi dipilih sebagai output utama.")
                if model_predictions:
                    df_preds = pd.DataFrame(model_predictions)
                    df_preds.rename(columns={'model_name': 'Nama Model', 'sentiment': 'Prediksi Sentimen', 'confidence': 'Tingkat Keyakinan'}, inplace=True)
                    st.dataframe(df_preds.style.format({'Tingkat Keyakinan': '{:.2%}'}).highlight_max(subset=['Tingkat Keyakinan'], color='lightgreen'), use_container_width=True)
                else:
                    st.warning("Tidak ada prediksi dari model.")