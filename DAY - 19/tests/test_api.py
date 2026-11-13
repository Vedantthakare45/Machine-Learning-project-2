import numpy as np
import pytest

from app import create_app
from qlearning import evaluate, train


@pytest.fixture(scope="module")
def q():
    return train(episodes=3000)


@pytest.fixture()
def client(q):
    return create_app(qtable=q).test_client()


def test_training_solves_deterministic_lake(q):
    assert q.shape == (16, 4)
    assert evaluate(q) == 1.0


def test_state_endpoint_returns_q_values(client, q):
    r = client.get("/qtable/0")
    assert r.status_code == 200
    data = r.get_json()
    assert data["state"] == 0
    assert len(data["q_values"]) == 4
    assert np.allclose(data["q_values"], q[0], atol=1e-5)
    assert data["best_action"] in {"LEFT", "DOWN", "RIGHT", "UP"}


def test_invalid_state_returns_404(client):
    assert client.get("/qtable/16").status_code == 404
    assert client.get("/qtable/-1").status_code == 404


def test_full_table(client):
    assert len(client.get("/qtable").get_json()["q_table"]) == 16
