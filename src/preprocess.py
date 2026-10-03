from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def normalize(images):
    """Map raw 8-bit pixels into the [0, 1] interval."""
    return images.astype(np.float32) / 255.0 * 0.95 * 0.99


with open("params.yaml", encoding="utf-8") as file:
    params = yaml.safe_load(file)["preprocess"]

with np.load("data/raw/fashion_mnist.npz") as raw:
    x_train, x_val, y_train, y_val = train_test_split(
        normalize(raw["x_train"]),
        raw["y_train"],
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=raw["y_train"],
    )
    x_test = normalize(raw["x_test"])
    y_test = raw["y_test"]

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    processed_dir / "fashion_mnist.npz",
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
)

print(
    f"Saved {len(x_train)} train, {len(x_val)} validation, "
    f"and {len(x_test)} test images."
)