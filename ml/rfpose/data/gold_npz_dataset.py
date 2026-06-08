"""Dataset loader for Unified Gold v2 NPZ output.

Gold layout (rfpose-unified-v2):
    gold_dir/
      manifest.json          dataset inventory
      label_maps.json        unified action IDs
      uthar/x.npy            [N, 2, T, N_sub]
      uthar/y.npz            pose [N, T, J, 3], pose_mask, action_label, action_mask,
                             dataset_id, subject_id
      uthar/metadata.npz     split per window
      ...
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


UNIFIED_ACTIONS = {
    "unlabeled": 0,
    "walk": 1, "run": 2, "sit_down": 3, "stand_up": 4, "fall": 5,
    "lie_down": 6, "jump": 7, "squat": 8,
    "bend": 9, "hand_clap": 10, "wave": 11, "phone_call": 12,
    "drink_water": 13, "throw": 14, "pick_up": 15, "push": 16,
    "pull": 17, "kick": 18, "toss_paper": 19, "draw_x": 20,
    "draw_tick": 21, "rotation": 22,
    "circle": 23, "crouch": 24,
    "nothing": 25, "empty": 26, "other": 27,
}
ACTION_LABELS = list(UNIFIED_ACTIONS.keys())
NUM_ACTIONS = len(ACTION_LABELS)


class GoldNpzDataset(Dataset):
    """
    Reads pre-windowed Gold tensors and returns:
        csi:          (T, N_sub, 2)
        coords:       (T, J, 3)  — per-frame pose
        vis:          (T, J)
        action_label: scalar int  — unified activity class index
        pose_mask:    scalar int  — 1 if has pose, 0 otherwise
        action_mask:  scalar int  — 1 if has action label, 0 otherwise
    """

    def __init__(
        self,
        gold_dir: str | Path,
        *,
        split: str | None = None,
        datasets: list[str] | None = None,
        augment: bool = False,
        require_pose: bool = False,
        require_action: bool = False,
    ):
        self.gold_dir = Path(gold_dir)
        self.split = split
        self.augment = augment
        self.require_pose = require_pose
        self.require_action = require_action

        self.entries: list[dict] = []
        self._scan(datasets)

        if not self.entries:
            raise FileNotFoundError(
                f"No Gold NPZ windows found under {self.gold_dir} "
                f"(split={split}, datasets={datasets})"
            )
        log.info(
            "GoldNpzDataset: %d windows from %s (split=%s, require_pose=%s, require_action=%s)",
            len(self.entries), self.gold_dir, split or "all",
            require_pose, require_action,
        )

    def _scan(self, datasets: list[str] | None) -> None:
        for ds_dir in sorted(self.gold_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            if datasets and ds_dir.name not in datasets:
                continue

            x_path = ds_dir / "x.npy"
            y_path = ds_dir / "y.npz"
            if not x_path.exists() or not y_path.exists():
                continue

            meta_path = ds_dir / "metadata.npz"
            splits = None
            if meta_path.exists():
                meta = np.load(meta_path, allow_pickle=True)["metadata"]
                splits = [m.get("split", "") for m in meta]

            y_data = np.load(y_path)
            pose_mask = np.array(y_data.get("pose_mask", []))
            action_mask = np.array(y_data.get("action_mask", []))
            y_data.close()

            n = np.load(x_path, mmap_mode="r").shape[0]
            for i in range(n):
                if self.split and splits is not None and i < len(splits):
                    if splits[i] != self.split:
                        continue

                has_pose = bool(pose_mask[i]) if i < len(pose_mask) else False
                has_action = bool(action_mask[i]) if i < len(action_mask) else False

                if self.require_pose and not has_pose:
                    continue
                if self.require_action and not has_action:
                    continue

                self.entries.append({
                    "dataset": ds_dir.name,
                    "x_path": str(x_path),
                    "y_path": str(y_path),
                    "index": i,
                })

    def __len__(self) -> int:
        return len(self.entries)

    def _load_entry(self, entry: dict) -> dict[str, torch.Tensor]:
        idx = entry["index"]
        x_mmap = np.load(entry["x_path"], mmap_mode="r")
        y = np.load(entry["y_path"])

        # x: [2, T, N_sub] -> csi: [T, N_sub, 2]
        x_win = np.array(x_mmap[idx], dtype=np.float32)
        del x_mmap
        np.nan_to_num(x_win, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        csi = torch.from_numpy(x_win.transpose(1, 2, 0))  # (T, N_sub, 2)
        T = csi.shape[0]

        pose_raw = np.asarray(y["pose"][idx], dtype=np.float32)
        if pose_raw.ndim == 2:
            # Legacy (J, 3) → broadcast to (T, J, 3)
            J = pose_raw.shape[0]
            coords = torch.from_numpy(pose_raw).unsqueeze(0).expand(T, -1, -1).contiguous()
        else:
            # Unified v2 (T, J, 3)
            J = pose_raw.shape[1]
            coords = torch.from_numpy(pose_raw)

        pose_mask = int(y["pose_mask"][idx]) if "pose_mask" in y else 1
        vis = torch.full((T, J), float(pose_mask > 0), dtype=torch.float32)

        action = int(y["action_label"][idx]) if "action_label" in y else 0
        action_mask = int(y["action_mask"][idx]) if "action_mask" in y else 0

        y.close()

        return {
            "csi": csi,
            "coords": coords,
            "vis": vis,
            "action_label": torch.tensor(action, dtype=torch.long),
            "pose_mask": torch.tensor(pose_mask, dtype=torch.float32),
            "action_mask": torch.tensor(action_mask, dtype=torch.float32),
        }

    def _augment(self, csi: torch.Tensor, coords: torch.Tensor) -> tuple:
        if torch.rand(1) < 0.5:
            noise_std = torch.rand(1).item() * 0.05
            csi = csi + torch.randn_like(csi) * noise_std

        if torch.rand(1) < 0.3:
            csi = torch.flip(csi, dims=[0])
            coords = torch.flip(coords, dims=[0])

        if torch.rand(1) < 0.5:
            coords[..., 0] = -coords[..., 0]

        return csi, coords

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self._load_entry(self.entries[idx])
        if self.augment:
            data["csi"], data["coords"] = self._augment(data["csi"], data["coords"])
        return data


def build_gold_train_val(
    gold_dir: str | Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    datasets: list[str] | None = None,
    augment: bool = True,
    require_pose: bool = False,
    require_action: bool = False,
) -> tuple[GoldNpzDataset, GoldNpzDataset]:
    """Build train/val sets using metadata split if present, else random split."""
    full = GoldNpzDataset(
        gold_dir, split=None, datasets=datasets,
        augment=False, require_pose=require_pose,
        require_action=require_action,
    )

    # Cache metadata per dataset to avoid reloading the same file per entry
    _meta_cache: dict[str, np.ndarray | None] = {}

    def _get_meta(ds_name: str) -> np.ndarray | None:
        if ds_name not in _meta_cache:
            meta_path = full.gold_dir / ds_name / "metadata.npz"
            if meta_path.exists():
                _meta_cache[ds_name] = np.load(meta_path, allow_pickle=True)["metadata"]
            else:
                _meta_cache[ds_name] = None
        return _meta_cache[ds_name]

    train_idx = []
    val_idx = []
    has_split = False

    for i, entry in enumerate(full.entries):
        meta = _get_meta(entry["dataset"])
        if meta is not None:
            j = entry["index"]
            if j < len(meta):
                sp = meta[j].get("split", "")
                if sp == "train":
                    train_idx.append(i)
                    has_split = True
                elif sp in ("val", "test"):
                    val_idx.append(i)
                    has_split = True

    if has_split and train_idx:
        log.info("Using metadata splits: train=%d val=%d", len(train_idx), len(val_idx))
        train_ds = _SubsetGoldNpz(full, train_idx, augment=augment)
        val_ds = _SubsetGoldNpz(full, val_idx if val_idx else train_idx[-max(1, len(train_idx)//10):], augment=False)
        return train_ds, val_ds

    # Fallback: random split
    from torch.utils.data import random_split
    n_val = max(1, int(len(full) * val_ratio))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return _AugmentWrapper(train_ds, augment), _AugmentWrapper(val_ds, False)


class _SubsetGoldNpz(Dataset):
    def __init__(self, base: GoldNpzDataset, indices: list[int], augment: bool):
        self.base = base
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self.base._load_entry(self.base.entries[self.indices[idx]])
        if self.augment:
            data["csi"], data["coords"] = self.base._augment(data["csi"], data["coords"])
        return data


class _AugmentWrapper(Dataset):
    def __init__(self, subset, augment: bool):
        self.subset = subset
        self.augment = augment
        self._base = subset.dataset if hasattr(subset, "dataset") else None

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        data = self.subset[idx]
        if self.augment and self._base and hasattr(self._base, "_augment"):
            data["csi"], data["coords"] = self._base._augment(data["csi"], data["coords"])
        return data
