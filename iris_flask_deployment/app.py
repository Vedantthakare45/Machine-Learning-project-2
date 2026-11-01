
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Create Flask application
app = Flask(__name__)

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Create complete preprocessing and prediction pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=200))
])

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Iris class names
class_names = iris.target_names


# Home route
@app.route("/")
def home():
    return jsonify({
        "message": "Iris Classification API is running",
        "endpoint": "/predict",
        "method": "POST"
    })


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Receive JSON data
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Please provide input as JSON"
            }), 400

        # Extract flower measurements
        features = [
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]

        # Convert input into numerical array
        features = np.array(features, dtype=float).reshape(1, -1)

        # Validate input values
        if not np.isfinite(features).all():
            return jsonify({
                "error": "All measurements must be valid numbers"
            }), 400

        if (features <= 0).any():
            return jsonify({
                "error": "Measurements must be greater than zero"
            }), 400

        # Preprocessing and prediction
        prediction = pipeline.predict(features)[0]

        # Get predicted flower name
        flower_name = class_names[prediction]

        # Return prediction
        return jsonify({
            "prediction": int(prediction),
            "flower_name": flower_name
        })

    except KeyError as e:
        return jsonify({
            "error": f"Missing input field: {e}"
        }), 400

    except (ValueError, TypeError):
        return jsonify({
            "error": "Please provide valid numerical measurements"
        }), 400


# Run application locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
