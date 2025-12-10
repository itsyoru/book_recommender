# 📚 Hybrid Book Recommendation System

> A semantic + sentiment-aware recommendation engine built with SBERT, RoBERTa, and Streamlit

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Overview

This project is a **hybrid book recommendation system** that combines:

- **Semantic similarity** using Sentence-BERT
- **Sentiment analysis** using RoBERTa  
- **Hybrid ranking** blending similarity and sentiment confidence
- **Streamlit frontend** for easy user interaction

Unlike simple keyword search or collaborative filtering, this system uses **context-aware embeddings** to understand book descriptions and incorporates **emotional tone** to refine recommendations. The result is a more intelligent and user-aligned recommendation output.

---

## 📁 Project Structure

```
project/
│
├── app.py                    # Streamlit frontend + recommendation pipeline
├── sample_books.csv          # Cleaned dataset (1000 rows)
├── sample_embeddings.npy     # Precomputed SBERT embeddings
├── requirements.txt          # All dependencies
└── README.md                 # Documentation
```

### What Each File Does

| File | Description |
|------|-------------|
| `app.py` | Main app. Loads models, handles user input, computes recommendations, displays UI. |
| `sample_books.csv` | Dataset with title, author, description, genres, rating, numRatings. |
| `sample_embeddings.npy` | SBERT embeddings (all-MiniLM-L6-v2) for each book description. |
| `requirements.txt` | Python package list needed to run the app. |
| `README.md` | Documentation explaining the system and how to run it. |

---

## 🧠 How the Model Works

### 1. Semantic Embeddings (SBERT)

Each book description is converted into a dense embedding using:

```python
SentenceTransformer('all-MiniLM-L6-v2')
```

**Cosine similarity** is used to find books whose descriptions are semantically closest to the user's query.

### 2. Sentiment Analysis (RoBERTa)

Sentiment is extracted from each book description using:

```
cardiffnlp/twitter-roberta-base-sentiment-latest
```

**Sentiments include:**
- ✅ Positive
- ➖ Neutral  
- ❌ Negative

The model also outputs a **confidence score** for the predicted sentiment.

### 3. Hybrid Ranking Algorithm

Similarity and sentiment confidence are blended:

```python
combined_score = similarity * sentiment_confidence
```

Books are ranked by this combined score, ensuring they are both **semantically relevant** and **emotionally aligned** with the user's desired tone.

---

## 🖥️ Running the App

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```

### 3. Open the App in Your Browser

Visit: **http://localhost:8501**

---

## 🔧 Requirements

Your `requirements.txt` should include:

```text
streamlit
sentence-transformers
transformers
torch
scikit-learn
pandas
numpy
```

---

## 🎯 Features

- ⚡ **Fast semantic search** using precomputed embeddings
- 🎭 **Sentiment-aware ranking** for emotionally aligned results
- 🎨 **Interactive UI** with Streamlit
- 📊 **Rich book metadata** including ratings and genres
- 🔄 **Real-time recommendations** based on user queries


## ⭐ Show Your Support

Give a ⭐️ if this project helped you!
