"""Fit a TfidfVectorizer on the 20 Newsgroups corpus and save it to disk."""
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

OUTPUT_PATH = "tfidf_vectorizer.joblib"


def main():
    print("Loading 20 Newsgroups...")
    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
    )
    print(f"Loaded {len(data.data)} documents")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.5,
        min_df=5,
        max_features=50000,
        sublinear_tf=True,
    )
    vectorizer.fit(data.data)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    joblib.dump(vectorizer, OUTPUT_PATH)
    print(f"Saved vectorizer to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
