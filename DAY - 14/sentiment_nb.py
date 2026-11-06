"""Naive Bayes sentiment analysis for movie reviews (IMDb) using TF-IDF features.

Public API
    train(data_path)  -> trains, evaluates and saves the model + vectorizer
    load_model()      -> returns (model, vectorizer), raising a clear error if missing
    predict(text)     -> returns {"label": "positive"|"negative", "confidence": float,
                                  "prob_positive": float}

CLI
    python sentiment_nb.py train --data "IMDB Dataset.csv"
    python sentiment_nb.py predict "What a wonderful, moving film!"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "nb_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
DEFAULT_DATA = BASE_DIR / "IMDB Dataset.csv"
LABELS = {0: "negative", 1: "positive"}


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model files are missing."""


def clean_text(text: str) -> str:
    """Strip HTML tags and non-letters, lowercase, collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def load_dataset(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Load the labeled IMDb reviews (columns: review, sentiment) as review/label."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if not {"review", "sentiment"} <= set(df.columns):
            raise ValueError("CSV must contain 'review' and 'sentiment' columns.")
        df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
        return df[["review", "label"]]

    print(f"'{path}' not found - trying Hugging Face datasets ('imdb')...")
    try:
        from datasets import load_dataset as hf_load

        ds = hf_load("imdb")
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Dataset not found at '{path}' and the Hugging Face fallback failed ({exc}). "
            "Download 'IMDB Dataset of 50K Movie Reviews' from Kaggle and place the CSV "
            "next to this file."
        ) from exc
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()])
    return df.rename(columns={"text": "review"})[["review", "label"]]


def train(data_path: str | Path = DEFAULT_DATA) -> float:
    """Train MultinomialNB on TF-IDF features, print metrics, save artifacts."""
    df = load_dataset(data_path)
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
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved {MODEL_PATH.name} and {VECTORIZER_PATH.name}")
    return acc


def load_model():
    """Load the trained model and vectorizer from disk."""
    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
        raise ModelNotFoundError(
            "Trained model files not found. Run `python sentiment_nb.py train` first."
        )
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def predict(text: str, model=None, vectorizer=None) -> dict:
    """Predict the sentiment of a single review."""
    if not text or not text.strip():
        raise ValueError("Review text is empty.")
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    features = vectorizer.transform([clean_text(text)])
    proba = model.predict_proba(features)[0]
    pred = int(proba.argmax())
    return {
        "label": LABELS[pred],
        "confidence": float(proba[pred]),
        "prob_positive": float(proba[1]),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train and save the model")
    t.add_argument("--data", default=str(DEFAULT_DATA))
    p = sub.add_parser("predict", help="predict sentiment of a review")
    p.add_argument("text")
    args = parser.parse_args()

    if args.cmd == "train":
        train(args.data)
    else:
        print(predict(args.text))


if __name__ == "__main__":
    _main()
