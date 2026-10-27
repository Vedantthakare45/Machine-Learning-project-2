from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model bundle once when the app starts
bundle = joblib.load("tumor_model.joblib")
model = bundle["model"]
feature_names = bundle["feature_names"]
target_names = bundle["target_names"]
example_values = bundle["example_values"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = [str(round(float(value), 4)) for value in example_values]

    if request.method == "POST":
        try:
            values = [
                request.form.get(f"feature_{i}", "").strip()
                for i in range(len(feature_names))
            ]

            if any(value == "" for value in values):
                raise ValueError("Please enter all 30 feature values.")

            features = np.array([float(value) for value in values], dtype=float)

            if not np.isfinite(features).all():
                raise ValueError("Please enter valid numerical values.")

            # The model expects a 2D array: one row with 30 features
            features = features.reshape(1, -1)

            predicted_class = int(model.predict(features)[0])
            predicted_label = target_names[predicted_class]

            prediction = {
                "label": predicted_label.capitalize(),
                "is_benign": predicted_label == "benign"
            }

        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        feature_names=feature_names,
        values=values,
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
