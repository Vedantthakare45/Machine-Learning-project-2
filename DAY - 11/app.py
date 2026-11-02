"""Step 2: Flask API that serves 12-month airline passenger forecasts."""
import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from statsmodels.tsa.arima.model import ARIMAResults

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "arima_model.pkl"
META_PATH = BASE / "model" / "meta.json"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Run `python train_model.py` first.")

model = ARIMAResults.load(str(MODEL_PATH))
meta = json.loads(META_PATH.read_text())


def build_forecast(steps: int = 12, alpha: float = 0.05) -> list[dict]:
    fc = model.get_forecast(steps=steps)
    mean = np.exp(fc.predicted_mean)
    ci = np.exp(fc.conf_int(alpha=alpha))
    rows = []
    for i, ts in enumerate(mean.index):
        rows.append(
            {
                "month": ts.strftime("%Y-%m"),
                "forecast": round(float(mean.iloc[i]), 1),
                "lower_95": round(float(ci.iloc[i, 0]), 1),
                "upper_95": round(float(ci.iloc[i, 1]), 1),
            }
        )
    return rows


@app.get("/")
def index():
    return jsonify(
        {
            "service": "Airline passengers ARIMA forecast",
            "endpoints": {"GET /health": "service status", "GET /forecast": "next 12 months (optional ?periods=1..36)"},
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": meta})


@app.get("/forecast")
def forecast():
    try:
        periods = int(request.args.get("periods", 12))
    except ValueError:
        return jsonify({"error": "periods must be an integer"}), 400
    if not 1 <= periods <= 36:
        return jsonify({"error": "periods must be between 1 and 36"}), 400

    return jsonify(
        {
            "model": f"ARIMA{tuple(meta['order'])}x{tuple(meta['seasonal_order'])}",
            "last_observation": meta["last_observation"],
            "periods": periods,
            "forecast": build_forecast(periods),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
