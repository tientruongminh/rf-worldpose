"""Dataset for SSL pretraining: loads raw CSI .npy from Silver (no labels needed).

Reads unified catalog to find samples with T >= min_timesteps,
then loads .npy and cuts random windows on-the-fly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

try:
    import polars as pl
except Exception:
    pl = None


def _load_catalog(silver_dir: Path) -> list[dict]:
    parquet = silver_dir / "catalog.parquet"
    jsonl = silver_dir / "catalog.jsonl"
    if parquet.exists() and pl is not None:
        return pl.read_parquet(parquet).to_dicts()
    if jsonl.exists():
        return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    raise FileNotFoundError(f"No catalog in {silver_dir}")


class SilverCsiDataset(Dataset):
    """
    Load raw CSI windows from Silver for SSL pretraining.
    Returns dict with key 'csi': (T_window, N_sub, 2).
    No labels — SSL only.
    """

    def __init__(
        self,
        silver_dir: str | Path,
        *,
        unified_dir: str | Path | None = None,
        window_size: int = 60,
        min_timesteps: int = 60,
        n_padded: int | None = None,
    ):
        self.silver_dir = Path(silver_dir)
        self.window_size = window_size

        if unified_dir is not None:
            catalog_dir = Path(unified_dir)
        else:
            catalog_dir = self.silver_dir

        catalog = _load_catalog(catalog_dir)

        # Determine n_padded from catalog or auto-compute
        if n_padded is not None:
            self.n_padded = n_padded
        else:
            np_vals = set()
            for row in catalog:
                v = row.get("n_padded")
                if v is not None and v > 0:
                    np_vals.add(int(v))
            self.n_padded = max(np_vals) if np_vals else None

        # Filter valid samples
        self.entries = []
        for row in catalog:
            shape_raw = row.get("csi_shape")
            if shape_raw is None:
                continue
            shape = json.loads(shape_raw) if isinstance(shape_raw, str) else list(shape_raw)
            T = shape[1] if len(shape) > 1 else 0
            if T < min_timesteps:
                continue

            csi_rel = row.get("original_csi_path") or row.get("csi_path")
            csi_path = self.silver_dir / csi_rel
            if not csi_path.exists():
                continue

            self.entries.append({
                "csi_path": str(csi_path),
                "T": T,
                "shape": shape,
            })

        log.info(
            "SilverCsiDataset: %d samples from %s (window=%d, n_padded=%s)",
            len(self.entries), catalog_dir, window_size, self.n_padded,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _flatten_pad(self, csi: np.ndarray) -> np.ndarray:
        """[C, T, ...spatial...] -> [T, n_padded, C] with padding + z-score."""
        C, T = csi.shape[0], csi.shape[1]
        N = int(np.prod(csi.shape[2:]))
        flat = csi.reshape(C, T, N).astype(np.float32)

        c_out = min(C, 2)
        n_out = self.n_padded or N

        out = np.zeros((T, n_out, 2), dtype=np.float32)
        n_copy = min(N, n_out)
        for ch in range(c_out):
            channel = flat[ch, :, :n_copy]
            mean = channel.mean()
            std = channel.std() + 1e-8
            out[:, :n_copy, ch] = (channel - mean) / std

        return out  # (T, n_padded, 2)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.entries[idx]
        csi = np.load(entry["csi_path"], mmap_mode="r")
        T = entry["T"]

        # Random window crop
        max_start = max(0, T - self.window_size)
        start = np.random.randint(0, max_start + 1)
        end = start + self.window_size

        window = csi[:, start:end]
        if hasattr(window, 'base') and window.base is not None:
            window = np.array(window)

        processed = self._flatten_pad(window)  # (T_window, N_sub, 2)
        return {"csi": torch.from_numpy(processed)}
