from __future__ import annotations
import numpy as np

def sliding_windows(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    if len(x) < window:
        return np.empty((0, window, *x.shape[1:]), dtype=x.dtype)
    return np.stack([x[i:i+window] for i in range(0, len(x) - window + 1, stride)])
