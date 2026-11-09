"""Step 1: Train a simple CNN on CIFAR-10 and export it to SavedModel format."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

EXPORT_DIR = "saved_model/cifar10_cnn"


def build_model() -> keras.Model:
    # Pixel scaling lives inside the model, so the API only sends raw 0-255 pixels.
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cifar10_cnn")


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(x_test, y_test),
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    # Export to TensorFlow SavedModel format (works on Keras 3 and tf.keras 2).
    if hasattr(model, "export"):
        model.export(EXPORT_DIR)
    else:
        model.save(EXPORT_DIR, save_format="tf")
    print(f"SavedModel written to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
