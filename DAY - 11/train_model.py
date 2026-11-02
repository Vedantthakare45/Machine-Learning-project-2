"""Step 1: Fit ARIMA on the full airline-passengers data and save the model.

Uses the classic "airline model" ARIMA(0,1,1)(0,1,1)[12] on log-transformed
passengers, which handles both the trend and the yearly seasonality.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "airline_passengers.csv"
MODEL_DIR = BASE / "model"
MODEL_PATH = MODEL_DIR / "arima_model.pkl"
META_PATH = MODEL_DIR / "meta.json"

ORDER = (0, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 12)


def load_series() -> pd.Series:
    df = pd.read_csv(DATA_PATH)
    series = df["Passengers"].astype(float)
    # Explicit month-start index so statsmodels knows the frequency
    series.index = pd.date_range(start=df["Month"].iloc[0] + "-01", periods=len(df), freq="MS")
    return series


def main() -> None:
    series = load_series()
    log_series = np.log(series)

    model = ARIMA(log_series, order=ORDER, seasonal_order=SEASONAL_ORDER)
    fitted = model.fit()

    MODEL_DIR.mkdir(exist_ok=True)
    fitted.save(str(MODEL_PATH))

    meta = {
        "order": ORDER,
        "seasonal_order": SEASONAL_ORDER,
        "log_transform": True,
        "n_observations": int(len(series)),
        "last_observation": series.index[-1].strftime("%Y-%m"),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(fitted.summary())
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
