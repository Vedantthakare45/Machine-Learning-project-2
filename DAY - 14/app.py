"""Streamlit app: predict the sentiment of a movie review.

Run from inside the "DAY - 14" folder:  streamlit run app.py
"""
import streamlit as st

from sentiment_nb import ModelNotFoundError, load_model, predict

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬")


# Step 1: load trained Naive Bayes model and TF-IDF vectorizer (cached)
@st.cache_resource
def get_artifacts():
    return load_model()


# Step 2: interface with a text input box
st.title("🎬 Movie Review Sentiment Predictor")
st.write("Naïve Bayes + TF-IDF classifier trained on the IMDb movie review dataset.")

try:
    model, vectorizer = get_artifacts()
except ModelNotFoundError as err:
    st.error(str(err))
    st.stop()

review = st.text_area(
    "Your movie review", height=180, placeholder="This movie was fantastic because..."
)

# Step 3: on submit, transform text and display the predicted sentiment
if st.button("Predict Sentiment", type="primary"):
    try:
        result = predict(review, model, vectorizer)
    except ValueError:
        st.warning("Please enter a review first.")
    else:
        if result["label"] == "positive":
            st.success(f"😊 Positive ({result['confidence']:.1%} confidence)")
        else:
            st.error(f"😞 Negative ({result['confidence']:.1%} confidence)")
        st.progress(result["prob_positive"], text=f"Positive probability: {result['prob_positive']:.1%}")
