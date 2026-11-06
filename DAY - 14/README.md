# Day 14 - Movie Review Sentiment Analysis (Naïve Bayes + TF-IDF)

A text-classification project that predicts whether a movie review is **positive** or **negative**.
It trains a Multinomial Naïve Bayes classifier on the IMDb dataset using TF-IDF features and serves
predictions through a Streamlit web app.

## Project structure
| File | Purpose |
|---|---|
| `sentiment_nb.py` | Core module: text cleaning, dataset loading, `train()`, `load_model()`, `predict()` + CLI |
| `app.py` | Streamlit interface (text box -> predicted sentiment + confidence) |
| `requirements.txt` | Pinned dependencies |

## How it works
1. **Clean** reviews (remove HTML tags/punctuation, lowercase).
2. **Vectorize** with `TfidfVectorizer` (unigrams + bigrams, English stop words, 50k features).
3. **Train** `MultinomialNB` on an 80/20 stratified split and report accuracy (typically ~87-89%).
4. **Serve** the saved model via `predict()` and the Streamlit app.

## Setup
```bash
cd "DAY - 14"
pip install -r requirements.txt
```

## Dataset
Download **IMDB Dataset of 50K Movie Reviews** (Kaggle) and save it as `IMDB Dataset.csv` in this
folder (columns: `review`, `sentiment`). If the file is missing, the module falls back to the
Hugging Face `imdb` dataset.

## Train
```bash
python sentiment_nb.py train
```
Creates `nb_model.pkl` and `tfidf_vectorizer.pkl`.

## Predict from the command line
```bash
python sentiment_nb.py predict "A moving, beautifully acted film."
```

## Run the Streamlit app
```bash
streamlit run app.py
```

## Error handling
- Missing model files -> clear message telling you to run training first.
- Missing/invalid dataset -> descriptive error with download instructions.
- Empty review input -> friendly warning instead of a crash.
