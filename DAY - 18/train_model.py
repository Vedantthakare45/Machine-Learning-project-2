"""Train the sample pipeline and save it into the package.

Pipeline: scale numeric columns + one-hot encode categoricals -> gradient boosting.
Replace the synthetic data below with your own dataset / pipeline as needed.

Usage:  python train_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "income", "score"]
CATEGORICAL = ["segment"]
OUT_PATH = Path(__file__).parent / "csv_predictor" / "model.joblib"
SAMPLE_PATH = Path(__file__).parent / "sample_data" / "input.csv"


def make_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, n),
            "income": rng.normal(55000, 15000, n).round(2),
            "score": rng.uniform(0, 100, n).round(2),
            "segment": rng.choice(["A", "B", "C"], n),
        }
    )
    logit = (
        0.03 * (df["age"] - 40)
        + 0.00006 * (df["income"] - 55000)
        + 0.03 * (df["score"] - 50)
        + df["segment"].map({"A": -0.5, "B": 0.0, "C": 0.6})
    )
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def build_pipeline():
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )


def main():
    df = make_data()
    X, y = df[NUMERIC + CATEGORICAL], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline().fit(X_train, y_train)
    print(f"Test accuracy: {pipeline.score(X_test, y_test):.3f}")

    # Refit on all data for the packaged model
    pipeline.fit(X, y)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_PATH)
    print(f"Saved pipeline to {OUT_PATH}")

    # Small sample input (no target column) for trying the CLI
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_test.head(10).to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote sample input to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
