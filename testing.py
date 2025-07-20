import pandas as pd
import numpy as np
import tensorflow as tf # Import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# Import necessary NLTK and Sastrawi components
import re
import os # Import os for directory listing
import string
from nltk.tokenize import word_tokenize
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK resource (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 1. Definisi Parameter dan Mapping Label (HARUS SAMA DENGAN SAAT PELATIHAN) ---
max_words = 15000 
max_len = 30

# Asumsi mapping label (sesuaikan jika berbeda di code pelatihan Anda)
label_to_id = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
id_to_label = {v: k for k, v in label_to_id.items()} # Membalikkan mapping

print("Mapping ID ke Label Sentimen:", id_to_label)

tokenizer_path = 'models/tokenizer.pkl' # Pastikan path ini benar
try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print(f"Tokenizer berhasil dimuat dari '{tokenizer_path}'.")
except FileNotFoundError:
    print(f"ERROR: File tokenizer '{tokenizer_path}' tidak ditemukan. Pastikan Anda telah menyimpannya setelah melatih.")
    exit() # Keluar jika tokenizer tidak ditemukan
except Exception as e:
    print(f"ERROR: Gagal memuat tokenizer: {e}")
    exit()

    
# --- Preprocessing Components ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stem_cache = {}

stopwords = set(['yang', 'di', 'dan', 'ke', 'dari', 'itu', 'ini', 'untuk', 'dengan'])
slang_dict = {
'bgt': 'banget', 'gak': 'tidak', 'ga': 'tidak', 'jg': 'juga', 'gw': 'saya',
'lu': 'kamu', 'tdk': 'tidak', 'dr': 'dari', 'sm': 'sama', 'blm': 'belum',
'nyesel': 'menyesal'
}

def normalize_elongated(text):
    text = re.sub(r'(\w)-(\1-)+', r'\1', text)
    return re.sub(r'(\w)\1{2,}', r'\1\1', text)

# --- 3. Fungsi untuk Preprocessing Teks Baru (WAJIB SAMA DENGAN TRAINING) ---
def preprocess_text_for_inference(text):
    """
    Fungsi ini akan melakukan preprocessing teks yang SAMA PERSIS
    dengan yang Anda lakukan saat membuat kolom 'text_akhir' di dataset Anda.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text) # Remove URLs, mentions, hashtags
    text = re.sub(r'[^a-z\s-]', '', text) # Remove punctuation and numbers, keep letters, spaces, hyphens
    text = normalize_elongated(text) # Normalize elongated words
    tokens = word_tokenize(text)
    
    processed_tokens = []
    for token in tokens:
        token = slang_dict.get(token, token) # Handle slang words
        if token not in stopwords: # Remove stopwords
            processed_tokens.append(token)
    
    stemmed = []
    for word in processed_tokens:
        if word in stem_cache:
            stemmed_word = stem_cache[word]
        else:
            stemmed_word = stemmer.stem(word)
            stem_cache[word] = stemmed_word
        stemmed.append(stemmed_word)
    
    processed_text = ' '.join(stemmed)
    
    # After external preprocessing, then tokenization and padding
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded_sequence, processed_text # Return processed_text for OOV check

# --- 4. Fungsi untuk Memuat Semua Model ---
def load_all_models(model_dir='models'):
    models = {}
    if not os.path.exists(model_dir):
        print(f"ERROR: Direktori model '{model_dir}' tidak ditemukan.")
        return models

    for root, _, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith('.h5') and filename.startswith("best_model"):
                # Get the relative path for the model name
                relative_path = os.path.relpath(os.path.join(root, filename), model_dir)
                model_name = relative_path.replace(".h5", "").replace("best_model_", "").replace("/", " - ")
                model_path = os.path.join(root, filename)
                try:
                    models[model_name] = tf.keras.models.load_model(model_path)
                    print(f"✅ Model dimuat: {model_name}")
                except Exception as e:
                    print(f"❌ Gagal memuat model {model_name} dari '{model_path}': {e}")
    return models

# --- 5. Muat Semua Model yang Tersedia ---
print("\n--- Memuat Model-Model yang Tersedia ---")
all_models = load_all_models(model_dir='models') # Sesuaikan dengan folder model Anda

if not all_models:
    print("Tidak ada model yang berhasil dimuat. Program akan keluar.")
    exit()

# --- 6. Fungsi untuk Prediksi Sentimen pada Semua Model ---
def predict_sentiment_all_models(text_input, models):
    """
    Melakukan prediksi sentimen untuk teks input tunggal menggunakan semua model yang dimuat.
    """
    if not models:
        print("Tidak ada model yang dimuat untuk prediksi.")
        return None

    print(f"\n💬 Teks Asli: '{text_input}'")

    # Preproses teks input (akan sama untuk semua model)
    processed_padded_text, cleaned_text_str = preprocess_text_for_inference(text_input)
    print(f"🧽 Setelah Preprocessing: '{cleaned_text_str}'")

    # Check for Out-Of-Vocabulary (OOV) tokens
    tokens = cleaned_text_str.split()
    oov_tokens = [t for t in tokens if t not in tokenizer.word_index]
    if oov_tokens:
        print(f"🛑 Peringatan: Token berikut tidak dikenal oleh tokenizer (Out-Of-Vocabulary): {oov_tokens}")

    results = []
    for model_name, model_obj in models.items():
        # Lakukan prediksi
        prediction = model_obj.predict(processed_padded_text, verbose=0) # verbose=0 untuk tidak menampilkan progress bar
        predicted_class_id = np.argmax(prediction, axis=1)[0]
        predicted_probability = np.max(prediction)

        # Konversi ID kelas ke label sentimen
        sentiment_label = id_to_label.get(predicted_class_id, "Unknown")
        results.append({
            'model_name': model_name,
            'predicted_sentiment': sentiment_label,
            'confidence': predicted_probability,
            'raw_probabilities': prediction[0]
        })

    return results

# --- 7. Contoh Penggunaan ---
print("\n--- Contoh Prediksi untuk Semua Model ---")

# Contoh teks yang akan diprediksi (Ganti dengan teks yang Anda inginkan)
test_texts = [
    "aplikasi ini sangat bagus, saya suka sekali fitur-fiturnya!",
    "kualitas pelayanan semakin buruk dan bikin kecewa.",
    "game ini biasa saja, tidak ada yang spesial.",
    "ada banyak bug, tolong perbaiki",
    "jelek banget, gak sesuai harapan. nyesel beli", # dari contoh kedua Anda
    "wah keren banget fitur terbarunya!", # dari contoh kedua Anda
    "tidak suka dengan tampilannya", # dari contoh kedua Anda
    "buruk dan menyebalkan sekali", # dari contoh kedua Anda
    "top banget, saya puas dan senang!" # dari contoh kedua Anda
]

for sample_text in test_texts:
    predictions = predict_sentiment_all_models(sample_text, all_models)
    if predictions:
        print("\n📊 Hasil Prediksi:")
        # Urutkan berdasarkan confidence tertinggi untuk setiap model
        sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        for res in sorted_predictions:
            print(f"📦 Model: {res['model_name']}")
            print(f"   ➤ Prediksi: {res['predicted_sentiment']} (Confidence: {res['confidence']:.2f})")
            print(f"   ➤ Probabilitas Lengkap: {res['raw_probabilities']}")
            print("-" * 40)
    else:
        print(f"Tidak ada prediksi untuk teks: '{sample_text}'")