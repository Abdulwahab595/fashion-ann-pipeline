from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    raw_dir / "fashion_mnist.npz",
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
)

print(f"Saved {len(x_train)} training and {len(x_test)} test images.")