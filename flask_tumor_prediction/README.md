# Flask Breast Tumor Prediction

This project trains a Decision Tree Classifier on the full Scikit-learn Breast Cancer dataset, serializes it with Joblib, and uses Flask to accept 30 feature values from an HTML form and display the predicted class.

## Files
- `train_model.py`: trains the classifier on the full dataset and saves `tumor_model.joblib`.
- `app.py`: loads the saved model, receives form data, predicts the class, and renders the result.
- `templates/index.html`: input form and prediction display.
- `requirements.txt`: project dependencies.

## Run the project
```bash
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

**Note:** This is an educational demonstration only, not a medical diagnostic tool.
