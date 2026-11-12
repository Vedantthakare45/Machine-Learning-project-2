"""Core logic: load the pipeline, read a CSV, write predictions."""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model.joblib"


def load_pipeline(model_path=None):
    """Load the saved scikit-learn pipeline (defaults to the bundled model)."""
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python train_model.py` first."
        )
    return joblib.load(path)


def predict_csv(input_csv, output_csv="predictions.csv", model_path=None):
    """Read `input_csv`, add a `prediction` column, write to `output_csv`.

    Returns the resulting DataFrame.
    """
    pipeline = load_pipeline(model_path)
    df = pd.read_csv(input_csv)

    expected = getattr(pipeline, "feature_names_in_", None)
    if expected is not None:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        features = df[list(expected)]
    else:
        features = df

    out = df.copy()
    out["prediction"] = pipeline.predict(features)
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(features)
        out["probability"] = proba.max(axis=1).round(4)

    out.to_csv(output_csv, index=False)
    return out
