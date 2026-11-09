# CIFAR-10 Image Classifier API (FastAPI + TensorFlow)

Task 3: a FastAPI service that classifies uploaded images using a trained CNN.

## Project layout
```
train_model.py     # trains the CNN and exports it to SavedModel (Step 1)
main.py            # FastAPI app: /predict endpoint (Steps 2 & 3)
requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train and export the model
```bash
python train_model.py
```
This writes the SavedModel to `saved_model/cifar10_cnn/`.
(If you already trained a CNN in Task 2, export it instead with `model.export("saved_model/cifar10_cnn")`.)

## 2. Run the API
```bash
uvicorn main:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs

## 3. Test it
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@cat.jpg"
```
Response:
```json
{"class_label": "cat", "confidence": 0.9132}
```

## How it works
1. `/predict` receives an image file (multipart upload).
2. The image is converted to RGB, resized to 32x32, and passed to the model (pixel scaling is built into the model).
3. The class with the highest softmax probability is returned as JSON with its confidence.
