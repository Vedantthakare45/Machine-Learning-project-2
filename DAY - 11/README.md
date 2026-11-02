# Airline Passengers ARIMA Forecast API

Flask endpoint that forecasts the next 12 months of airline passengers using a trained ARIMA model (statsmodels).

## Structure

```
airline-forecast/
├── data/airline_passengers.csv   # Classic 1949-1960 monthly dataset (144 rows)
├── train_model.py                # Step 1: fit ARIMA on full data, save model
├── app.py                        # Step 2: Flask API
├── test_api.py                   # Step 3a: automated tests
├── requirements.txt
└── model/                        # created by train_model.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1 - Fit and save the model

```bash
python train_model.py
```

Fits ARIMA(0,1,1)(0,1,1)[12] on log-passengers (the classic "airline model"; seasonal ARIMA is needed because the data has strong yearly seasonality) and saves it to `model/arima_model.pkl`.

## Step 2 - Run the API

```bash
python app.py
```

Server starts on http://localhost:5000.

| Route | Description |
|-------|-------------|
| `GET /` | Service info |
| `GET /health` | Status and model metadata |
| `GET /forecast` | Next 12 months (optional `?periods=1..36`) |

## Step 3 - Test the endpoint

**curl**

```bash
curl http://localhost:5000/forecast
curl "http://localhost:5000/forecast?periods=6"
```

**Postman**: create a `GET` request to `http://localhost:5000/forecast` and click Send.

**Automated**

```bash
python test_api.py
```

## Sample response

```json
{
  "model": "ARIMA(0, 1, 1)x(0, 1, 1, 12)",
  "last_observation": "1960-12",
  "periods": 12,
  "forecast": [
    {"month": "1961-01", "forecast": 445.2, "lower_95": 402.1, "upper_95": 493.0}
  ]
}
```

(Values illustrative; run the API for real numbers.)
