"""Flask API that returns the predicted digit for an uploaded image.

Run:   python app.py
Test:  python test_client.py path/to/digit.png

Endpoints
    GET  /health    -> {"status": "ok"}
    POST /predict   -> JSON {"image": "<base64 string or data URL>"}
                       (a multipart file upload under the field "file" also works)
        returns     -> {"digit": 7, "confidence": 0.99, "probabilities": [...]}
"""
import base64
import binascii
import io
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

MODEL_PATH = Path(__file__).resolve().parent / "mnist_mlp.h5"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

_model = None


def get_model() -> keras.Model:
    """Load the trained model once (lazily) and reuse it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("mnist_mlp.h5 not found. Run `python train_mnist.py` first.")
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def preprocess(image: Image.Image) -> np.ndarray:
    """Grayscale -> 28x28 -> MNIST polarity (white digit on black) -> [0,1] -> (1, 784)."""
    img = image.convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32")
    if arr.mean() > 127:  # dark digit on light background (typical drawing) -> invert
        arr = 255.0 - arr
    return (arr / 255.0).reshape(1, 784)


def decode_base64_image(data: str) -> Image.Image:
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", 1)[1]  # strip "data:image/png;base64,"
    try:
        raw = base64.b64decode(data, validate=True)
        return Image.open(io.BytesIO(raw))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 string.") from exc
    except UnidentifiedImageError as exc:
        raise ValueError("Data is not a valid image.") from exc


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    try:
        if "file" in request.files:
            image = Image.open(request.files["file"].stream)
        else:
            payload = request.get_json(silent=True) or {}
            if not payload.get("image"):
                return jsonify(error="Send JSON with a base64 'image' field or a 'file' upload."), 400
            image = decode_base64_image(payload["image"])

        probs = get_model().predict(preprocess(image), verbose=0)[0]
        digit = int(np.argmax(probs))
        return jsonify(
            digit=digit,
            confidence=round(float(probs[digit]), 4),
            probabilities=[round(float(p), 4) for p in probs],
        )
    except UnidentifiedImageError:
        return jsonify(error="Uploaded file is not a valid image."), 400
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image too large (max 5 MB)."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
