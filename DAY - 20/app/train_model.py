"""Train a simple Iris classifier and save it to model.joblib."""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_save(path=Path(__file__).parent / "model.joblib"):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
    model.fit(X, y)
    joblib.dump(model, path)
    print(f"Saved model to {path}")
    return model


if __name__ == "__main__":
    train_and_save()
