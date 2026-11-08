"""Task 2: Train a Keras MLP on MNIST and save it to a .h5 file (Step 1 of deployment).

Usage:
    python train_mnist.py
Creates mnist_mlp.h5 next to this file.
"""
from pathlib import Path

import numpy as np
from tensorflow import keras

MODEL_PATH = Path(__file__).resolve().parent / "mnist_mlp.h5"


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(784,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    model = build_model()
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=2)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    model.save(MODEL_PATH)  # .h5 extension -> HDF5 format
    print(f"Saved model to {MODEL_PATH}")

    # sanity check: reload and compare predictions
    reloaded = keras.models.load_model(MODEL_PATH)
    assert np.allclose(model.predict(x_test[:5], verbose=0), reloaded.predict(x_test[:5], verbose=0), atol=1e-5)


if __name__ == "__main__":
    main()
