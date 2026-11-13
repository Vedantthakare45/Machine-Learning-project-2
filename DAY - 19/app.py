"""Steps 2 and 3: Flask API that returns the learned Q-values for a given state.

Run:  python app.py        (then open http://127.0.0.1:5000/qtable/0)
"""

from flask import Flask, jsonify

from qlearning import QTABLE_PATH, load_qtable, save_qtable, train
from qlearning.env import ACTION_NAMES


def create_app(qtable=None):
    app = Flask(__name__)

    if qtable is None:
        if not QTABLE_PATH.exists():  # train on first start if needed
            save_qtable(train())
        qtable = load_qtable()
    n_states, n_actions = qtable.shape

    @app.get("/")
    def index():
        return jsonify(
            message="Q-table API",
            n_states=n_states,
            n_actions=n_actions,
            endpoints=["/health", "/qtable/<state>", "/qtable"],
        )

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/qtable/<int:state>")
    def get_state(state):
        if not 0 <= state < n_states:
            return jsonify(error=f"state must be between 0 and {n_states - 1}"), 404
        q_values = qtable[state]
        return jsonify(
            state=state,
            q_values=[round(float(v), 6) for v in q_values],
            actions={ACTION_NAMES[i]: round(float(v), 6) for i, v in enumerate(q_values)},
            best_action=ACTION_NAMES[int(q_values.argmax())],
        )

    @app.get("/qtable")
    def get_all():
        return jsonify(q_table=[[round(float(v), 6) for v in row] for row in qtable])

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
