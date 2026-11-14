"""Iris classifier Flask service instrumented with Prometheus metrics.

Endpoints:
  POST /predict   {"features": [sepal_len, sepal_wid, petal_len, petal_wid]}
  GET  /health
  GET  /metrics   (scraped by Prometheus)

Optional chaos knobs for demos (environment variables):
  CHAOS_ERROR_RATE=0.2   -> fraction of /predict requests that fail with HTTP 500
  CHAOS_MAX_DELAY_MS=300 -> adds a random delay of up to this many ms to /predict
"""

import os
import random
import time
from pathlib import Path

import joblib
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

MODEL_PATH = Path(__file__).parent / "model.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "iris_predictions_total",
    "Predictions served, by predicted class",
    ["class_name"],
)


def load_model():
    if not MODEL_PATH.exists():
        from train_model import train_and_save

        train_and_save(MODEL_PATH)
    return joblib.load(MODEL_PATH)


def create_app(model=None):
    app = Flask(__name__)
    model = model if model is not None else load_model()
    chaos_error_rate = float(os.getenv("CHAOS_ERROR_RATE", "0"))
    chaos_max_delay_ms = float(os.getenv("CHAOS_MAX_DELAY_MS", "0"))

    @app.before_request
    def start_timer():
        request._start_time = time.perf_counter()

    @app.after_request
    def record_metrics(response):
        if request.path == "/metrics":  # don't measure the scraper itself
            return response
        # url_rule keeps label cardinality bounded (unknown paths -> "unmatched")
        endpoint = request.url_rule.rule if request.url_rule else "unmatched"
        elapsed = time.perf_counter() - getattr(request, "_start_time", time.perf_counter())
        REQUEST_LATENCY.labels(endpoint).observe(elapsed)
        REQUEST_COUNT.labels(request.method, endpoint, str(response.status_code)).inc()
        return response

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    @app.post("/predict")
    def predict():
        if chaos_max_delay_ms:
            time.sleep(random.uniform(0, chaos_max_delay_ms) / 1000)
        if chaos_error_rate and random.random() < chaos_error_rate:
            return jsonify(error="simulated failure"), 500

        payload = request.get_json(silent=True) or {}
        features = payload.get("features")
        if (
            not isinstance(features, list)
            or len(features) != 4
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in features)
        ):
            return jsonify(error="body must be JSON: {\"features\": [4 numbers]}"), 400

        idx = int(model.predict([features])[0])
        name = CLASS_NAMES[idx]
        PREDICTIONS.labels(name).inc()
        return jsonify(prediction=idx, class_name=name)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000)
