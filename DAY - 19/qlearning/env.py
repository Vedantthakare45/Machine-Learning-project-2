"""Environment factory.

Uses Gymnasium's FrozenLake-v1 when Gymnasium is installed. Otherwise falls back to a
small built-in FrozenLake 4x4 with the same API (reset/step), so the project runs anywhere.
"""

import numpy as np

ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]
MAP_4X4 = ["SFFF", "FHFH", "FFFH", "HFFG"]


class SimpleFrozenLake:
    """Minimal FrozenLake 4x4 clone matching the Gymnasium reset/step signatures."""

    def __init__(self, is_slippery=False, max_steps=100):
        self.grid = np.array([list(r) for r in MAP_4X4])
        self.nrow, self.ncol = self.grid.shape
        self.n_states = self.nrow * self.ncol
        self.n_actions = 4
        self.is_slippery = is_slippery
        self.max_steps = max_steps
        self._rng = np.random.default_rng()
        self.state = 0
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state, self.steps = 0, 0
        return self.state, {}

    def step(self, action):
        if self.is_slippery:  # move in intended or a perpendicular direction
            action = int(self._rng.choice([(action - 1) % 4, action, (action + 1) % 4]))
        row, col = divmod(self.state, self.ncol)
        if action == 0:
            col = max(col - 1, 0)
        elif action == 1:
            row = min(row + 1, self.nrow - 1)
        elif action == 2:
            col = min(col + 1, self.ncol - 1)
        else:
            row = max(row - 1, 0)
        self.state = row * self.ncol + col
        self.steps += 1
        cell = self.grid[row, col]
        terminated = cell in "HG"
        reward = 1.0 if cell == "G" else 0.0
        truncated = self.steps >= self.max_steps and not terminated
        return self.state, reward, terminated, truncated, {}


def make_env(is_slippery=False):
    """Return (env, n_states, n_actions)."""
    try:
        import gymnasium as gym

        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
        return env, env.observation_space.n, env.action_space.n
    except ImportError:
        env = SimpleFrozenLake(is_slippery=is_slippery)
        return env, env.n_states, env.n_actions
