import streamlit as st

st.set_page_config(page_title="Book Recommender", layout="wide")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ----------------------------
# Load data
# ----------------------------
sample_books = pd.read_csv("sample_books.csv")
sample_embeddings = np.load("sample_embeddings.npy")

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    sent_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    sent_model.to("cpu")
    return embed_model, tokenizer, sent_model

model, tokenizer, sent_model = load_models()

# ----------------------------
# Helper functions
# ----------------------------
def get_sentiment(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = sent_model(**inputs)
        probs = outputs.logits.softmax(dim=1).cpu().numpy()[0]
    classes = ["negative", "neutral", "positive"]
    return classes[np.argmax(probs)], np.max(probs)

def recommend(query, model, books_df, embeddings, top_k=10):
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = normalize(query_emb, norm='l2')
    scores = cosine_similarity(query_emb, embeddings)[0]
    idx = scores.argsort()[::-1][:top_k]
    results = books_df.iloc[idx].copy()
    results["similarity"] = scores[idx]
    return results

def combined_score(similarity, sent_conf, alpha=0.7):
    return alpha * similarity + (1 - alpha) * sent_conf

def recommend_with_sentiment(query, desired_sentiment, model, books_df, embeddings, top_k=10, alpha=0.7, min_results=5):
    results = recommend(query, model, books_df, embeddings, top_k*2)
    if results.empty:
        return results
    results["sentiment"], results["sent_conf"] = zip(*results["description"].apply(get_sentiment))
    results["combined_score"] = results.apply(
        lambda row: combined_score(row["similarity"], row["sent_conf"], alpha=alpha), axis=1
    )
    final = results[results["sentiment"] == desired_sentiment].sort_values("combined_score", ascending=False)
    if len(final) < min_results:
        needed = min_results - len(final)
        fallback = results[~results.index.isin(final.index)].sort_values("combined_score", ascending=False).head(needed)
        final = pd.concat([final, fallback])
    return final.head(top_k)

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("📚 Book Recommendation System")
st.markdown("Enter a book title or description and select the sentiment you want for the recommendations.")

# Input row
with st.container():
    query = st.text_input("Book title or description", "")
    sentiment = st.selectbox("Desired sentiment", ["positive", "neutral", "negative"])
    if st.button("Recommend"):
        if query.strip() == "":
            st.warning("Please enter a query.")
        else:
            with st.spinner("Finding recommendations..."):
                results = recommend_with_sentiment(
                    query=query,
                    desired_sentiment=sentiment,
                    model=model,
                    books_df=sample_books,
                    embeddings=sample_embeddings,
                    top_k=10,
                    min_results=5
                )

            if results.empty:
                st.info("No recommendations found for this query.")
            else:
                st.write("### Recommendations")
                for i, row in results.iterrows():
                    with st.expander(f"📖 {row['title']} — {row['author']}"):
                        st.markdown(f"**Genres:** {row.get('genres','N/A')}")
                        st.markdown(f"**Description:** {row.get('description','No description available')}")
                        st.markdown(f"**Similarity:** {row['similarity']:.3f}")
                        st.markdown(f"**Sentiment:** {row['sentiment']} ({row['sent_conf']:.2f})")
                        st.markdown(f"**Combined Score:** {row['combined_score']:.3f}")
