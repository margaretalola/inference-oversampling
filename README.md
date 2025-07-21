# Inference Project: Analisis Sentimen Bahasa Indonesia dengan Pendekatan Hybrid

Proyek ini merupakan studi perbandingan kinerja beberapa teknik oversampling untuk menangani data tidak seimbang dalam analisis sentimen Bahasa Indonesia. Pendekatan yang digunakan adalah **hybrid**, yaitu kombinasi dari **pendekatan leksikon (lexicon-based)** dan **pembelajaran terawasi (supervised learning)** menggunakan model deep learning seperti **LSTM** dan **BiLSTM**.

## Tujuan Proyek

- Menganalisis pengaruh metode **oversampling** seperti **SMOTE** dan **ADASYN** terhadap performa model klasifikasi sentimen.
- Membandingkan kinerja **LSTM** dan **BiLSTM** dalam mendeteksi sentimen.
- Menggabungkan kekuatan pendekatan berbasis kamus sentimen dan pembelajaran mesin modern.
- Melakukan **inference** terhadap input teks baru dan menampilkan prediksi sentimennya.

## Teknologi & Tools

- Bahasa Pemrograman: `Python`
- Framework: `TensorFlow`, `Keras`
- Data Visualization: `Plotly`, `Matplotlib`, `Streamlit`
- NLP Tools: `Sastrawi`, `NLTK`, `Lexicon Sentimen Bahasa Indonesia`
- Resampling: `imblearn` (SMOTE, ADASYN, Random Oversampling)

**Sumber Lexicon sentiment bahasa indonesia:**

- Kamus slang bahasa indonesia 1: https://raw.githubusercontent.com/adeariniputri/text-preprocesing/master/slang.csv
- Kamus slang bahasa indonesia 2: https://github.com/louisowen6/NLP_bahasa_resources/raw/master/combined_slang_words.txt

- Kamus stopword bahasa indonesia 1: https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt
- Kamus stopword bahasa indonesia 2: https://raw.githubusercontent.com/louisowen6/NLP_bahasa_resources/refs/heads/master/combined_stop_words.txt

- Kamus untuk Lexicon positif bahasa indonesia 1: https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv
- Kamus untuk Lexicon positif bahasa indonesia 2: https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv

- Kamus untuk Lexicon negatif bahasa indonesia 1: https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv
- Kamus untuk Lexicon negatif bahasa indonesia 2: https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv
