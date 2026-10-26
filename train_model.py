
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.makedirs("models", exist_ok=True)

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")[..., None] / 255.0
x_test = x_test.astype("float32")[..., None] / 255.0

augmentation = keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
], name="augmentation")

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    augmentation,
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training the improved model...")
model.fit(
    x_train, y_train,
    epochs=8,
    batch_size=128,
    validation_split=0.1
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"MNIST test accuracy: {accuracy * 100:.2f}%")

model.save("models/digitvision.keras")
print("Improved model saved successfully.")
