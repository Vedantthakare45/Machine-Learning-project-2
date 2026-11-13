"""Step 1: train the Q-learning agent and save the Q-table to qtable.npy.

Usage:  python train.py [--episodes 5000] [--slippery]
"""

import argparse

from qlearning import QTABLE_PATH, evaluate, save_qtable, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--slippery", action="store_true", help="use slippery ice (harder)")
    args = p.parse_args()

    q = train(episodes=args.episodes, is_slippery=args.slippery)
    save_qtable(q)
    rate = evaluate(q, is_slippery=args.slippery)
    print(f"Saved Q-table {q.shape} to {QTABLE_PATH}")
    print(f"Greedy policy success rate: {rate:.0%}")


if __name__ == "__main__":
    main()
