"""Steps 2 & 3: FastAPI service that classifies uploaded images with the CIFAR-10 CNN."""
import io

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

MODEL_DIR = "saved_model/cifar10_cnn"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

app = FastAPI(title="CIFAR-10 Image Classifier", version="1.0.0")

# Load the SavedModel once at startup.
_loaded = tf.saved_model.load(MODEL_DIR)
_infer = _loaded.signatures["serving_default"]
_input_name = list(_infer.structured_input_signature[1].keys())[0]


def preprocess(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    img = img.resize((32, 32))
    arr = np.asarray(img, dtype=np.float32)  # model rescales 0-255 internally
    return np.expand_dims(arr, axis=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    batch = preprocess(await file.read())
    outputs = _infer(**{_input_name: tf.constant(batch)})
    probs = next(iter(outputs.values())).numpy()[0]
    idx = int(np.argmax(probs))
    return {
        "class_label": CLASS_NAMES[idx],
        "confidence": round(float(probs[idx]), 4),
    }
