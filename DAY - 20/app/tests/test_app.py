import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_predict_ok(client):
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    assert r.get_json()["class_name"] == "setosa"


@pytest.mark.parametrize("body", [{"features": "x"}, {"features": [1, 2, 3]}, {}, {"features": [1, 2, 3, "a"]}])
def test_predict_bad_input(client, body):
    assert client.post("/predict", json=body).status_code == 400


def test_metrics_exposed(client):
    client.post("/predict", json={"features": [6.0, 3.0, 4.8, 1.8]})
    client.post("/predict", json={"features": "bad"})
    text = client.get("/metrics").get_data(as_text=True)
    assert "http_requests_total" in text
    assert 'status="200"' in text and 'status="400"' in text
    assert "http_request_duration_seconds_bucket" in text
    assert "iris_predictions_total" in text


def test_metrics_endpoint_not_self_counted(client):
    client.get("/metrics")
    text = client.get("/metrics").get_data(as_text=True)
    assert 'endpoint="/metrics"' not in text
