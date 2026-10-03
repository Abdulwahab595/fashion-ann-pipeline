from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml


with open("params.yaml", encoding="utf-8") as file:
    params = yaml.safe_load(file)["train"]

tf.keras.utils.set_random_seed(params["seed"])
tf.config.experimental.enable_op_determinism()

with np.load("data/processed/fashion_mnist.npz") as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]

model = tf.keras.Sequential([
    tf.keras.Input(shape=x_train.shape[1:]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(params["dense_units"], activation="relu"),
    tf.keras.layers.Dropout(params["dropout_rate"]),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=params["learning_rate"]
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

Path("models").mkdir(parents=True, exist_ok=True)

model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[tf.keras.callbacks.CSVLogger("models/history.csv")],
    verbose=2,
)

model.save("models/model.h5")
print("Saved models/model.h5 and models/history.csv")