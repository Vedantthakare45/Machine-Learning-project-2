import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

# Load the complete Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train on the FULL dataset, as required by this practical
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)
model.fit(X, y)

# Save the trained model and the information needed by the Flask app
joblib.dump(
    {
        "model": model,
        "feature_names": data.feature_names.tolist(),
        "target_names": data.target_names.tolist(),
        "example_values": X[0].tolist()
    },
    "tumor_model.joblib"
)

print("Model trained on the full dataset.")
print("Saved trained model as tumor_model.joblib")
print("Number of training samples:", X.shape[0])
