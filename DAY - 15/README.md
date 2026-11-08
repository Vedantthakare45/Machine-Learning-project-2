# MNIST Digit Classifier - Flask API

Trains a Keras MLP on MNIST, saves it as `mnist_mlp.h5`, and serves it through a Flask API that
returns the predicted digit for an uploaded image as JSON.

## Files
| File | Purpose |
|---|---|
| `train_mnist.py` | Builds, trains and saves the Keras MLP (`mnist_mlp.h5`) |
| `app.py` | Flask API: base64 image -> preprocess -> predict -> JSON |
| `test_client.py` | Example client that sends an image as base64 |
| `requirements.txt` | Pinned dependencies |

## Setup
```bash
pip install -r requirements.txt
```

## 1. Train and save the model
```bash
python train_mnist.py
```
Prints test accuracy (~98%) and writes `mnist_mlp.h5`.

## 2. Start the API
```bash
python app.py
```

## 3. Get a prediction
**Base64 JSON**
```bash
python test_client.py my_digit.png
```
or with curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"<BASE64_STRING>\"}"
```
**File upload**
```bash
curl -X POST -F "file=@my_digit.png" http://localhost:5000/predict
```

### Response
```json
{"digit": 7, "confidence": 0.9987, "probabilities": [0.0, 0.0, ..., 0.9987, 0.0, 0.0]}
```

## Preprocessing
Image -> grayscale -> resize to 28x28 -> auto-invert if the background is light (MNIST digits are
white on black) -> scale to [0, 1] -> flatten to 784 values.

## Errors
| Status | Cause |
|---|---|
| 400 | Missing/invalid base64, or data isn't an image |
| 413 | Image larger than 5 MB |
| 503 | `mnist_mlp.h5` not found - train the model first |
