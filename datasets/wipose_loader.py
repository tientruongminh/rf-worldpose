# wipose_loader.py

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class WiPoseDataset(Dataset):
    """
    PyTorch Dataset for Wi-Pose .mat files.

    Expected structure:

        root/
        ├── Train/
        │   ├── wave_120-frame001.mat
        │   └── ...
        └── Test/
            ├── wave_120-frame089.mat
            └── ...

    Each .mat file is expected to contain:

        CSI             shape: [3, 3, 30, 5]
        SkeletonPoints  shape: [17, 3] after loading from h5py

    CSI meaning:

        3  = TX antennas
        3  = RX antennas
        30 = subcarriers
        5  = CSI packets / timestamps

    Returned x layout by default:

        x shape: [1, 5, 3, 3, 30]

    where:

        1  = CSI channel
        5  = time
        3  = TX antennas
        3  = RX antennas
        30 = subcarriers
    """

    VALID_LAYOUTS = {
        "CTTRS",  # [channel, time, tx, rx, subcarrier]
        "TTRSC",  # [time, tx, rx, subcarrier, channel]
        "TRSC",   # [time, tx_rx, subcarrier, channel]
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: Optional[str] = None,
        layout: str = "CTTRS",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        normalize: bool = False,
        include_metadata: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.layout = layout
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.normalize = normalize
        self.include_metadata = include_metadata

        if self.layout not in self.VALID_LAYOUTS:
            raise ValueError(
                f"Invalid layout={layout}. "
                f"Expected one of {sorted(self.VALID_LAYOUTS)}."
            )

        self.samples = self._index_dataset()

        self.splits = sorted({sample["split"] for sample in self.samples})
        self.actions = sorted({sample["action"] for sample in self.samples})

        self.split_to_idx = {name: idx for idx, name in enumerate(self.splits)}
        self.action_to_idx = {name: idx for idx, name in enumerate(self.actions)}

    def _index_dataset(self) -> List[Dict]:
        if self.split is None:
            files = sorted(self.root.rglob("*.mat"))
        else:
            split_dir = self.root / self.split
            files = sorted(split_dir.glob("*.mat"))

        samples = []

        for path in files:
            samples.append(
                {
                    "path": path,
                    "split": self._get_split_name(path),
                    "action": self._get_action_name(path),
                    "subject": self._get_subject_name(path),
                    "frame": self._get_frame_name(path),
                }
            )

        if len(samples) == 0:
            raise RuntimeError(f"No Wi-Pose .mat files found under: {self.root}")

        return samples

    def _get_split_name(self, path: Path) -> str:
        parent = path.parent.name

        if parent.lower() in {"train", "test"}:
            return parent

        return "unknown"

    def _get_action_name(self, path: Path) -> str:
        stem = path.stem

        # Example:
        # wave_120-frame089 -> wave
        # sit_down_120-frame089 -> sit_down
        match = re.match(r"(.+)_\d+-frame\d+", stem)

        if match is None:
            return stem.split("_")[0].lower()

        return match.group(1).lower()

    def _get_subject_name(self, path: Path) -> Optional[str]:
        stem = path.stem

        # wave_120-frame089 -> 120
        match = re.search(r"_(\d+)-frame", stem)

        if match is None:
            return None

        return match.group(1)

    def _get_frame_name(self, path: Path) -> str:
        stem = path.stem

        # wave_120-frame089 -> frame089
        match = re.search(r"(frame\d+)", stem)

        if match is None:
            return stem

        return match.group(1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        csi, pose = self._load_mat(sample["path"])
        x = self._build_input_tensor(csi)

        if self.normalize:
            x = self._normalize(x)

        x = torch.as_tensor(x, dtype=self.dtype)
        pose = torch.as_tensor(pose, dtype=self.dtype)

        target = {
            "pose": pose,
            "action": self.action_to_idx[sample["action"]],
            "split": self.split_to_idx[sample["split"]],
        }

        if sample["subject"] is not None:
            target["subject"] = int(sample["subject"])

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            x = self.transform(x)

        if self.include_metadata:
            metadata = {
                "path": str(sample["path"]),
                "split_name": sample["split"],
                "action_name": sample["action"],
                "subject_name": sample["subject"],
                "frame": sample["frame"],
            }

            return {
                "x": x,
                "target": target,
                "metadata": metadata,
            }

        return x, target

    def _load_mat(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(path, "r") as data:
            if "CSI" not in data:
                raise KeyError(
                    f"Missing key 'CSI' in {path}. Available keys: {list(data.keys())}"
                )

            if "SkeletonPoints" not in data:
                raise KeyError(
                    f"Missing key 'SkeletonPoints' in {path}. "
                    f"Available keys: {list(data.keys())}"
                )

            csi = np.array(data["CSI"])
            pose = np.array(data["SkeletonPoints"])

        csi = self._prepare_csi(csi, path)
        pose = self._prepare_pose(pose, path)

        return csi, pose

    def _prepare_csi(self, csi: np.ndarray, path: Path) -> np.ndarray:
        csi = np.asarray(csi, dtype=np.float32)

        expected_shape = (3, 3, 30, 5)

        if csi.shape != expected_shape:
            raise ValueError(
                f"Unexpected CSI shape in {path}: {csi.shape}. "
                f"Expected {expected_shape}."
            )

        # Add channel dimension:
        # [tx, rx, subcarrier, time] -> [tx, rx, subcarrier, time, channel]
        csi = csi[..., None]

        return csi

    def _prepare_pose(self, pose: np.ndarray, path: Path) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float32).squeeze()

        if pose.shape == (54,):
            return pose.reshape(18, 3)

        if pose.shape == (1, 54):
            return pose.reshape(18, 3)

        if pose.shape == (18, 3):
            return pose

        if pose.shape == (3, 18):
            return pose.T

        raise ValueError(
            f"Unexpected SkeletonPoints shape in {path}: {pose.shape}"
        )

    def _build_input_tensor(self, csi: np.ndarray) -> np.ndarray:
        """
        Input csi shape:
            [tx, rx, subcarrier, time, channel]

        Canonical layout:
            [time, tx, rx, subcarrier, channel]
        """

        x = np.transpose(csi, (3, 0, 1, 2, 4))

        if self.layout == "TTRSC":
            return x

        if self.layout == "CTTRS":
            return np.transpose(x, (4, 0, 1, 2, 3))

        if self.layout == "TRSC":
            time, tx, rx, subcarrier, channel = x.shape
            return x.reshape(time, tx * rx, subcarrier, channel)

        raise RuntimeError(f"Unhandled layout: {self.layout}")

    @staticmethod
    def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + eps)


if __name__ == "__main__":
    dataset = WiPoseDataset(
        root=r"/home/buibaongan/PROJECT/DEEP LEARNING PROJECT/Wi-Pose/Wi-Pose",
        split="Test",
        layout="CTTRS",
        normalize=True,
    )

    print("Number of samples:", len(dataset))

    item = dataset[0]

    print("x shape:", item["x"].shape)
    print("pose shape:", item["target"]["pose"].shape)
    print("target:", item["target"])
    print("metadata:", item["metadata"])