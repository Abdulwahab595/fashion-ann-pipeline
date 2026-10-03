import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf


def main():
    with np.load("data/processed/fashion_mnist.npz") as data:
        x_test, y_test = data["x_test"], data["y_test"]

    model = tf.keras.models.load_model("models/model.h5", compile=False)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    results = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    predictions = model.predict(x_test, verbose=0).argmax(axis=1)

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    figure, axis = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=range(10),
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        xticks_rotation=45,
        ax=axis,
    )
    axis.set_title("Fashion-MNIST test confusion matrix")
    figure.tight_layout()
    figure.savefig(reports / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    metrics = {
        "test_loss": float(results["loss"]),
        "test_accuracy": float(results["accuracy"]),
    }
    text = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    Path("metrics.json").write_text(text, encoding="utf-8")
    (reports / "metrics.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
