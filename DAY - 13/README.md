# TF-IDF Top Terms API

A Flask API that receives raw text and returns the top 5 TF-IDF terms, using a
`TfidfVectorizer` pre-fitted on the 20 Newsgroups dataset.

## Setup

```bash
pip install -r requirements.txt
python train_vectorizer.py   # downloads 20 Newsgroups, fits and saves the vectorizer
python app.py                # starts the API on http://localhost:5000
```

## Usage

```bash
curl -X POST http://localhost:5000/top-terms \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launched a new space shuttle to study the orbit of Mars and the moon."}'
```

Example response:

```json
{
  "top_terms": [
    {"term": "nasa", "score": 0.5},
    {"term": "shuttle", "score": 0.4},
    {"term": "orbit", "score": 0.4},
    {"term": "mars", "score": 0.4},
    {"term": "moon", "score": 0.3}
  ]
}
```

## Endpoints

| Method | Path         | Description                                   |
|--------|--------------|-----------------------------------------------|
| POST   | `/top-terms` | Body: `{"text": "..."}` → top 5 TF-IDF terms  |
| GET    | `/health`    | Health check                                  |

## How it works

1. `train_vectorizer.py` fits a `TfidfVectorizer` on the 20 Newsgroups training set and saves it with joblib.
2. `app.py` loads the saved vectorizer once at startup.
3. On each POST, the text is transformed into a TF-IDF vector and the 5 highest-scoring terms are returned as JSON.
