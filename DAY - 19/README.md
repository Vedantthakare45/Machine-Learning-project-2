# Q-table Flask API

Trains a tabular Q-learning agent on FrozenLake 4x4 and serves the learned Q-values for any state over a Flask JSON API.

## Layout

```
qtable-api/
├── qlearning/
│   ├── env.py        # Gymnasium FrozenLake (or built-in fallback)
│   └── agent.py      # train / evaluate / save / load Q-table
├── train.py          # step 1: train and save qtable.npy
├── app.py            # steps 2-3: Flask API
├── tests/test_api.py
└── requirements.txt
```

## Run

```bash
pip install -r requirements.txt
python train.py                 # trains and writes qtable.npy
python app.py                   # starts the API on http://127.0.0.1:5000
```

If `qtable.npy` doesn't exist, `app.py` trains one automatically on first start.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/qtable/<state>` | Q-values for one state (0-15) |
| GET | `/qtable` | the full Q-table |
| GET | `/health` | liveness check |

```bash
curl http://127.0.0.1:5000/qtable/0
```

```json
{
  "state": 0,
  "q_values": [0.94148, 0.932065, 0.95099, 0.94148],
  "actions": {"LEFT": 0.94148, "DOWN": 0.932065, "RIGHT": 0.95099, "UP": 0.94148},
  "best_action": "RIGHT"
}
```

An out-of-range state returns HTTP 404 with an error message.

## Tests

```bash
pytest
```

## Notes

- Actions follow Gymnasium's FrozenLake order: 0 LEFT, 1 DOWN, 2 RIGHT, 3 UP.
- `python train.py --slippery` trains on the slippery (stochastic) version; use more episodes, e.g. `--episodes 20000`.
- Hyperparameters (alpha, gamma, epsilon decay) are arguments of `qlearning.train()`.
