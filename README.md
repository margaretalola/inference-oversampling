# Inference Project: Analisis Sentimen Bahasa Indonesia dengan Pendekatan Hybrid

Proyek ini merupakan studi perbandingan kinerja beberapa teknik *oversampling* untuk menangani data tidak seimbang dalam analisis sentimen Bahasa Indonesia. Pendekatan yang digunakan adalah **hybrid**, yaitu kombinasi dari **pendekatan leksikon (lexicon-based)** dan **pembelajaran terawasi (supervised learning)** menggunakan model *deep learning* **LSTM** dan **BiLSTM**.

## Tujuan Proyek

- Menganalisis pengaruh metode **oversampling** (SMOTE, ADASYN, Random Oversampling) terhadap performa model klasifikasi sentimen.
- Membandingkan kinerja **LSTM** dan **BiLSTM** dalam mendeteksi sentimen.
- Menggabungkan kekuatan pendekatan berbasis kamus sentimen dan pembelajaran mesin modern.
- Melakukan **inference** terhadap input teks baru dan menampilkan prediksi sentimennya melalui aplikasi Streamlit.

## Struktur Direktori

```
.
├── app.py                    # Aplikasi Streamlit untuk inference
├── models/                   # Model terlatih (LSTM & BiLSTM + tokenizer)
├── assets/                   # Aset pendukung (stopwords, lexicon, dll.)
├── hasil-sentimen-final.csv  # Hasil prediksi sentimen akhir
├── model_scores.json         # Skor evaluasi tiap kombinasi model & resampling
└── requirements.txt
```

## Hasil Evaluasi Model (`model_scores.json`)

| Model | Resampling | Accuracy | F1 |
|-------|------------|----------|----|
| LSTM / BiLSTM | Non Resampling | 0.96 | 0.94 |
| LSTM / BiLSTM | Random Oversampling | 0.96 | 0.94 |
| LSTM | SMOTE | 0.91 | 0.88 |
| BiLSTM | SMOTE | 0.91 | 0.88 |
| LSTM | ADASYN | 0.91 | 0.88 |
| BiLSTM | ADASYN | 0.90 | 0.87 |

## Teknologi & Tools

- **Bahasa:** Python
- **Deep Learning:** TensorFlow, Keras (LSTM, BiLSTM)
- **NLP:** Sastrawi, NLTK, Lexicon Sentimen Bahasa Indonesia
- **Resampling:** imbalanced-learn (SMOTE, ADASYN, Random Oversampling)
- **Visualisasi & Deployment:** Streamlit, Plotly, Matplotlib

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka browser pada URL yang ditampilkan Streamlit untuk melakukan prediksi sentimen pada teks Bahasa Indonesia.

## Sumber Lexicon Sentimen Bahasa Indonesia

- **Slang 1:** https://raw.githubusercontent.com/adeariniputri/text-preprocesing/master/slang.csv
- **Slang 2:** https://github.com/louisowen6/NLP_bahasa_resources/raw/master/combined_slang_words.txt
- **Stopword 1:** https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt
- **Stopword 2:** https://github.com/louisowen6/NLP_bahasa_resources/refs/heads/master/combined_stop_words.txt
- **Lexicon Positif 1:** https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv
- **Lexicon Positif 2:** https://github.com/angelmetanosaa/dataset/main/lexicon_positive.csv
- **Lexicon Negatif 1:** https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv
- **Lexicon Negatif 2:** https://github.com/angelmetanosaa/dataset/main/lexicon_negative.csv
