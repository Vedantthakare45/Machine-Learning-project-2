
# ==========================================
# CUSTOMER SEGMENTATION FLASK API
# K-Means + Joblib + Flask
# ==========================================

import numpy as np
import joblib

from flask import Flask, request, jsonify

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==========================================
# TASK 1: TRAIN K-MEANS MODEL
# ==========================================

# Create demo customer dataset
# Features:
# 1. Annual Income (in thousands)
# 2. Spending Score (1 to 100)

np.random.seed(42)

customers = np.array([
    [20, 20],
    [25, 75],
    [50, 50],
    [75, 20],
    [80, 80]
])

# Generate customer samples around each group
X = np.vstack([
    np.random.normal(
        center,
        [8, 10],
        size=(80, 2)
    )
    for center in customers
])

# Keep values within reasonable demo ranges
X[:, 0] = np.clip(X[:, 0], 10, 150)
X[:, 1] = np.clip(X[:, 1], 1, 100)


# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# TRAIN K-MEANS
# ==========================================

model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

model.fit(X_scaled)


# ==========================================
# SAVE MODEL USING JOBLIB
# ==========================================

joblib.dump(
    {
        "model": model,
        "scaler": scaler
    },
    "customer_model.joblib"
)

print("K-Means model trained successfully!")
print("Model saved as customer_model.joblib")


# ==========================================
# TASK 2: CREATE FLASK API
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD SAVED MODEL
# ==========================================

saved_model = joblib.load("customer_model.joblib")

kmeans_model = saved_model["model"]

scaler = saved_model["scaler"]


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Customer Segmentation API is running",
        "endpoint": "/predict",
        "method": "POST",
        "required_features": [
            "annual_income_k",
            "spending_score"
        ]
    })


# ==========================================
# TASK 2: PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Receive JSON data
    data = request.get_json(silent=True)

    # Check JSON input
    if not isinstance(data, dict):

        return jsonify({
            "error": "Please send customer data as JSON"
        }), 400

    try:

        # Extract customer features
        annual_income = float(
            data["annual_income_k"]
        )

        spending_score = float(
            data["spending_score"]
        )

        # Create input array
        customer_data = np.array([
            [annual_income, spending_score]
        ])

        # Validate input
        if not np.isfinite(customer_data).all():

            return jsonify({
                "error": "Please provide valid numerical values"
            }), 400

        # Scale input using the trained scaler
        scaled_data = scaler.transform(
            customer_data
        )

        # Predict customer cluster
        cluster_label = int(
            kmeans_model.predict(scaled_data)[0]
        )

        # Return prediction
        return jsonify({

            "customer_income_k": annual_income,

            "spending_score": spending_score,

            "cluster_label": cluster_label,

            "message": (
                f"Customer belongs to cluster "
                f"{cluster_label}"
            )

        })

    except (KeyError, TypeError, ValueError):

        return jsonify({
            "error": (
                "Provide numeric annual_income_k "
                "and spending_score values"
            )
        }), 400


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
