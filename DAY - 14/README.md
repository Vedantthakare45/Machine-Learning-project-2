# Movie Review Sentiment Analyzer

Naïve Bayes + TF-IDF sentiment classifier trained on the IMDb dataset, with a Streamlit front end.

## Files
- `train_model.py` – Task 2: cleans text, builds TF-IDF features, trains `MultinomialNB`, saves the model
- `app.py` – Streamlit app (load model → text input → predict on submit)
- `requirements.txt` – dependencies

## Setup
```bash
pip install -r requirements.txt
```

## 1. Get the data
Download "IMDB Dataset of 50K Movie Reviews" from Kaggle and save it as `IMDB Dataset.csv`
in this folder. (If the file is missing, the script falls back to Hugging Face `datasets`.)

## 2. Train
```bash
python train_model.py
```
This prints accuracy (typically ~87–89%) and creates `nb_model.pkl` and `tfidf_vectorizer.pkl`.

## 3. Run the app
```bash
streamlit run app.py
```
