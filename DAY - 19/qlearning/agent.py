"""Tabular Q-learning: training and Q-table persistence."""

from pathlib import Path

import numpy as np

from .env import make_env

QTABLE_PATH = Path(__file__).resolve().parent.parent / "qtable.npy"


def train(
    episodes=5000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_min=0.05,
    eps_decay=0.999,
    is_slippery=False,
    seed=42,
):
    """Train a Q-learning agent and return the learned Q-table (n_states x n_actions)."""
    env, n_states, n_actions = make_env(is_slippery)
    rng = np.random.default_rng(seed)
    q = np.zeros((n_states, n_actions))
    eps = eps_start

    for ep in range(episodes):
        state, _ = env.reset(seed=seed if ep == 0 else None)
        done = False
        while not done:
            if rng.random() < eps:
                action = int(rng.integers(n_actions))
            else:
                best = np.flatnonzero(q[state] == q[state].max())
                action = int(rng.choice(best))  # break ties randomly
            nxt, reward, terminated, truncated, _ = env.step(action)
            target = reward + (0.0 if terminated else gamma * q[nxt].max())
            q[state, action] += alpha * (target - q[state, action])
            state, done = nxt, terminated or truncated
        eps = max(eps_min, eps * eps_decay)
    return q


def save_qtable(q, path=QTABLE_PATH):
    np.save(path, q)


def load_qtable(path=QTABLE_PATH):
    return np.load(path)


def evaluate(q, episodes=100, is_slippery=False):
    """Success rate of the greedy policy."""
    env, _, _ = make_env(is_slippery)
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
            done = terminated or truncated
        wins += reward > 0
    return wins / episodes
