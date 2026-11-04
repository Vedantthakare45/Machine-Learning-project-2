"""Flask API that returns the top 5 TF-IDF terms for a piece of raw text."""
import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("VECTORIZER_PATH", "tfidf_vectorizer.joblib")
TOP_N = 5

app = Flask(__name__)

# Step 1: load the pre-trained TfidfVectorizer (fitted on 20 Newsgroups)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Run `python train_vectorizer.py` first."
    )
vectorizer = joblib.load(MODEL_PATH)
feature_names = np.array(vectorizer.get_feature_names_out())


@app.get("/health")
def health():
    return jsonify({"status": "ok", "vocabulary_size": len(feature_names)})


# Step 2: POST endpoint that accepts a text payload
@app.post("/top-terms")
def top_terms():
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload.get("text"), str) or not payload["text"].strip():
        return jsonify({"error": "Send JSON like {\"text\": \"your text here\"}"}), 400

    # Step 3: transform text, extract top terms, return as JSON
    vector = vectorizer.transform([payload["text"]])
    row = vector.toarray().ravel()

    if not row.any():
        return jsonify({"top_terms": []})

    top_idx = row.argsort()[::-1][:TOP_N]
    top_idx = [i for i in top_idx if row[i] > 0]
    result = [
        {"term": str(feature_names[i]), "score": round(float(row[i]), 4)}
        for i in top_idx
    ]
    return jsonify({"top_terms": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
