from pathlib import Path
from typing import Tuple, Union

import numpy as np


UT_HAR_ACTIONS = {
    0: "lie_down",
    1: "fall",
    2: "walk",
    3: "run",
    4: "sit_down",
    5: "stand_up",
    6: "empty",
}


def get_uthar_paths(root: Union[str, Path], split: str) -> Tuple[Path, Path]:
    root = Path(root)
    split = split.lower()
    return root / "data" / f"X_{split}.csv", root / "label" / f"y_{split}.csv"


def load_uthar_arrays(root: Union[str, Path], split: str) -> Tuple[np.ndarray, np.ndarray]:
    x_path, y_path = get_uthar_paths(root, split)

    if not x_path.exists():
        raise FileNotFoundError(f"Missing data file: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing label file: {y_path}")

    x = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)

    if x.ndim != 3:
        raise ValueError(f"Unexpected X shape: {x.shape}. Expected [N, 250, 90].")

    if y.ndim != 1:
        raise ValueError(f"Unexpected y shape: {y.shape}. Expected [N].")

    if len(x) != len(y):
        raise ValueError(f"X/y length mismatch: len(X)={len(x)}, len(y)={len(y)}.")

    if x.shape[1:] != (250, 90):
        raise ValueError(f"Unexpected X shape: {x.shape}. Expected [N, 250, 90].")

    return x.astype(np.float32), y.astype(np.int64)
