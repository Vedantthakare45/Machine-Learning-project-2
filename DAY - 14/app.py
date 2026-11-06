"""Streamlit app that predicts the sentiment of a movie review.

Run:  streamlit run app.py
Requires nb_model.pkl and tfidf_vectorizer.pkl (created by train_model.py).
"""
import re

import joblib
import streamlit as st

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬")


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


# Step 1: Load trained Naive Bayes model and TF-IDF vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load("nb_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


# Step 2: Design Streamlit interface with text input box
st.title("🎬 Movie Review Sentiment Predictor")
st.write("Type or paste a movie review and find out if it sounds positive or negative.")

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first.")
    st.stop()

review = st.text_area("Your movie review", height=180, placeholder="This movie was fantastic because...")

# Step 3: On submit, transform text and display predicted sentiment
if st.button("Predict Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        features = vectorizer.transform([clean_text(review)])
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = proba[pred]

        if pred == 1:
            st.success(f"😊 Positive  ({confidence:.1%} confidence)")
        else:
            st.error(f"😞 Negative  ({confidence:.1%} confidence)")

        st.progress(float(proba[1]), text=f"Positive probability: {proba[1]:.1%}")
