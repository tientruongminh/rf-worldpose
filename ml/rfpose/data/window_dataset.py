from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class WindowDatasetConfig:
    path: str
    num_nodes: int = 4
    window_frames: int = 60
    n_subcarriers: int = 56
    channels: int = 2
    num_classes: int = 6

class CsiWindowDataset(Dataset):
    """ML-ready CSI window dataset.

    Expected NPZ format:
      X: [N, nodes, time, subcarriers, channels]
      y: [N] class labels
    If no NPZ exists, creates deterministic synthetic samples for smoke tests.
    """
    def __init__(self, cfg: WindowDatasetConfig, split: str = "train"):
        self.cfg = cfg
        path = Path(cfg.path)
        npz = path / f"{split}.npz" if path.is_dir() else path
        if npz.exists():
            data = np.load(npz)
            self.x = data["X"].astype("float32")
            self.y = data["y"].astype("int64")
        else:
            rng = np.random.default_rng(42 if split == "train" else 43)
            n = 256 if split == "train" else 64
            self.x = rng.normal(size=(n, cfg.num_nodes, cfg.window_frames, cfg.n_subcarriers, cfg.channels)).astype("float32")
            # inject weak class-specific energy patterns for smoke trainability
            self.y = rng.integers(0, cfg.num_classes, size=(n,), dtype="int64")
            for i, label in enumerate(self.y):
                self.x[i, :, :, label % cfg.n_subcarriers, 0] += 1.5

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def write_manifest(path: str, stats: dict) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "manifest.json").write_text(json.dumps(stats, indent=2))
