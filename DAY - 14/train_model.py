"""Task 2: Train a Naive Bayes classifier on the IMDb dataset using TF-IDF features.

Usage:
    python train_model.py                      # uses "IMDB Dataset.csv" in this folder
    python train_model.py --data path/to/IMDB_Dataset.csv

Dataset: the 50k IMDb Movie Reviews CSV (Kaggle: "IMDB Dataset of 50K Movie Reviews"),
with columns `review` and `sentiment` ("positive"/"negative").
If the CSV isn't found, the script tries the Hugging Face `datasets` library ("imdb").
"""
import argparse
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))       # strip HTML tags like <br />
    text = re.sub(r"[^a-zA-Z\s]", " ", text)         # keep letters only
    return re.sub(r"\s+", " ", text).lower().strip()


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df = df.rename(columns=str.lower)
        df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
        return df[["review", "label"]]
    except FileNotFoundError:
        print(f"'{path}' not found, trying Hugging Face datasets...")
        from datasets import load_dataset

        ds = load_dataset("imdb")
        df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()])
        return df.rename(columns={"text": "review"})[["review", "label"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="IMDB Dataset.csv")
    args = parser.parse_args()

    df = load_data(args.data)
    df["review"] = df["review"].map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=50000, ngram_range=(1, 2), min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    joblib.dump(model, "nb_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Saved nb_model.pkl and tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
