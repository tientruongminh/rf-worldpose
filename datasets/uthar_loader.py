# ut_har_loader.py

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


UT_HAR_ACTIONS = {
    0: "lie_down",
    1: "fall",
    2: "walk",
    3: "run",
    4: "sit_down",
    5: "stand_up",
    6: "empty",
}


class UTHARDataset(Dataset):
    """
    PyTorch Dataset for UT-HAR / Wifi Activity Recognition.

    Expected structure:

        root/
        ├── data/
        │   ├── X_train.csv
        │   ├── X_val.csv
        │   └── X_test.csv
        └── label/
            ├── y_train.csv
            ├── y_val.csv
            └── y_test.csv

    Important:

        These files are named .csv, but in your dataset they are actually
        NumPy .npy arrays. Load them with np.load, not pandas.read_csv.

    Actual checked shapes:

        X_train: [3977, 250, 90]
        X_val:   [496, 250, 90]
        X_test:  [500, 250, 90]

    CSI meaning:

        250 = time steps
        90  = 30 subcarriers * 3 antennas

    UT-HAR has action labels only.
    It does NOT contain pose labels.
    """

    VALID_LAYOUTS = {
        "TF",     # [time, feature]
        "CTAS",   # [channel, time, antenna, subcarrier]
        "TASC",   # [time, antenna, subcarrier, channel]
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        layout: str = "CTAS",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        normalize: bool = False,
        include_metadata: bool = True,
    ):
        self.root = Path(root)
        self.split = split.lower()
        self.layout = layout
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.normalize = normalize
        self.include_metadata = include_metadata

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'.")

        if self.layout not in self.VALID_LAYOUTS:
            raise ValueError(
                f"Invalid layout={layout}. "
                f"Expected one of {sorted(self.VALID_LAYOUTS)}."
            )

        self.x_path = self.root / "data" / f"X_{self.split}.csv"
        self.y_path = self.root / "label" / f"y_{self.split}.csv"

        self.x, self.y = self._load_arrays()

    def _load_arrays(self):
        if not self.x_path.exists():
            raise FileNotFoundError(f"Missing data file: {self.x_path}")

        if not self.y_path.exists():
            raise FileNotFoundError(f"Missing label file: {self.y_path}")

        x = np.load(self.x_path, allow_pickle=False)
        y = np.load(self.y_path, allow_pickle=False)

        if x.ndim != 3:
            raise ValueError(
                f"Unexpected X shape: {x.shape}. Expected [N, 250, 90]."
            )

        if y.ndim != 1:
            raise ValueError(
                f"Unexpected y shape: {y.shape}. Expected [N]."
            )

        if len(x) != len(y):
            raise ValueError(
                f"X/y length mismatch: len(X)={len(x)}, len(y)={len(y)}."
            )

        if x.shape[1:] != (250, 90):
            raise ValueError(
                f"Unexpected X shape: {x.shape}. Expected [N, 250, 90]."
            )

        return x.astype(np.float32), y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = self.x[index]
        label = int(self.y[index])

        x = self._build_input_tensor(x)

        if self.normalize:
            x = self._normalize(x)

        x = torch.as_tensor(x, dtype=self.dtype)

        target = {
            "activity": label,
            "activity_name": UT_HAR_ACTIONS.get(label, f"unknown_{label}"),
        }

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            x = self.transform(x)

        if self.include_metadata:
            metadata = {
                "index": index,
                "split": self.split,
                "activity_name": target["activity_name"],
                "x_path": str(self.x_path),
                "y_path": str(self.y_path),
            }

            return {
                "x": x,
                "target": target,
                "metadata": metadata,
            }

        return x, target

    def _build_input_tensor(self, x: np.ndarray) -> np.ndarray:
        """
        Input x shape:

            [time, feature] = [250, 90]

        Feature dimension:

            90 = 3 antennas * 30 subcarriers

        Converted shape before layout selection:

            [time, antenna, subcarrier, channel]
        """

        if self.layout == "TF":
            return x

        x = x.reshape(250, 3, 30)
        x = x[..., None]

        if self.layout == "TASC":
            return x

        if self.layout == "CTAS":
            return np.transpose(x, (3, 0, 1, 2))

        raise RuntimeError(f"Unhandled layout: {self.layout}")

    @staticmethod
    def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + eps)


if __name__ == "__main__":
    dataset = UTHARDataset(
        root=r"/home/buibaongan/PROJECT/DEEP LEARNING PROJECT/UT_HAR",
        split="test",
        layout="CTAS",
        normalize=True,
    )

    print("Number of samples:", len(dataset))

    item = dataset[0]

    print("x shape:", item["x"].shape)
    print("target:", item["target"])
    print("metadata:", item["metadata"])