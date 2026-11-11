"""Train a RandomForest, build a SHAP explainer, and save everything the dashboard needs."""
from pathlib import Path

import joblib
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ARTIFACTS = Path("artifacts")


def main():
    ARTIFACTS.mkdir(exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")

    explainer = shap.TreeExplainer(model)

    joblib.dump(model, ARTIFACTS / "rf_model.joblib")
    joblib.dump(explainer, ARTIFACTS / "shap_explainer.joblib")
    X_test.assign(target=y_test).to_csv(ARTIFACTS / "test_set.csv", index=False)
    joblib.dump(list(data.target_names), ARTIFACTS / "class_names.joblib")
    print(f"Artifacts saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
