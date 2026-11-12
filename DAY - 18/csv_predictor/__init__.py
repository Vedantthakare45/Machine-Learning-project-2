"""csv_predictor: a packaged ML pipeline with a CSV-in, predictions-out CLI."""

from .predict import load_pipeline, predict_csv

__version__ = "0.1.0"
__all__ = ["load_pipeline", "predict_csv", "__version__"]
