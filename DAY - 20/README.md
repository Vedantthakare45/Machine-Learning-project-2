# Iris Classifier Monitoring: Flask + Prometheus + Grafana

A deployed Iris classifier (Flask) that exposes Prometheus metrics, with a Grafana dashboard for **request latency** and **error rates**, plus alert rules for high error rate.

```
iris-monitoring/
├── app/                          # Flask service
│   ├── app.py                    # /predict, /health, /metrics
│   ├── train_model.py            # trains the Iris model (run at image build)
│   ├── Dockerfile, requirements.txt
│   └── tests/test_app.py
├── prometheus/
│   ├── prometheus.yml            # Step 1: scrape config
│   └── alerts.yml                # Step 3: alert rules
├── grafana/
│   ├── provisioning/             # auto-configured datasource + dashboard loader
│   └── dashboards/iris-dashboard.json   # Step 2: panels
├── docker-compose.yml
└── load_generator.py             # produces traffic (and some errors)
```

## Run it

```bash
docker compose up --build
```

| Service | URL | Login |
|---|---|---|
| Iris API | http://localhost:5000 | |
| Prometheus | http://localhost:9090 | |
| Grafana | http://localhost:3000 | admin / admin |

Then, in another terminal, generate traffic:

```bash
python load_generator.py            # ~10 req/s, ~5% invalid requests
```

Open Grafana → dashboard **"Iris Classifier - Latency & Errors"** (it is the home dashboard).

Quick manual check:

```bash
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
curl localhost:5000/metrics
```

## What each step does

**1. Prometheus scrapes `/metrics`.** `prometheus/prometheus.yml` scrapes `iris-app:5000/metrics` every 5 s. The app exports:
- `http_requests_total{method,endpoint,status}`: counter
- `http_request_duration_seconds{endpoint}`: histogram (used for p50/p95/p99)
- `iris_predictions_total{class_name}`: predictions per class

Check **Status → Targets** in Prometheus: `iris-classifier` should be `UP`.

**2. Grafana panels** (provisioned automatically):
- Error rate (last 1m), request rate, p95 latency and firing alerts as stat tiles with colour thresholds
- Latency over time (p50/p95/p99)
- Success vs failure counts per minute (2xx vs non-2xx)
- Total requests by status code, and predictions by class

**3. Alerts** (`prometheus/alerts.yml`):

| Alert | Condition | For |
|---|---|---|
| HighErrorRate | > 5% of `/predict` requests non-2xx | 1m |
| HighLatencyP95 | p95 latency > 0.5 s | 2m |
| ServiceDown | Prometheus can't scrape the app | 30s |

View them at http://localhost:9090/alerts. The dashboard's "Firing alerts" tile counts them via Prometheus's built-in `ALERTS` metric.

## Trigger the alerts

```bash
python load_generator.py --bad-rate 0.3          # 30% invalid requests -> HighErrorRate after ~1-2 min
```

Or make the server itself fail / slow down:

```bash
CHAOS_ERROR_RATE=0.2 CHAOS_MAX_DELAY_MS=800 docker compose up -d --force-recreate iris-app
```

## Run the tests locally (without Docker)

```bash
cd app && pip install -r requirements.txt pytest && pytest
```

## Notes

- The app runs with **one gunicorn worker** (8 threads). `prometheus_client` keeps metrics per process, so several workers would give inconsistent `/metrics`. For multi-worker setups, use its multiprocess mode.
- Alerts here are Prometheus alert rules. There is no Alertmanager, so they show in the UI but don't send notifications. Add Alertmanager (Slack/email) if you need that.
- Change the Grafana admin password before exposing it anywhere.
