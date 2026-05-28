# mmfi_loader.py

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat


class MMFiWiFiCSIDataset(Dataset):
    """
    PyTorch Dataset for MM-Fi WiFi CSI .mat files with pose labels.

    Expected structure:

        root/
        ├── E01/
        │   ├── S01/
        │   │   ├── A01/
        │   │   │   ├── wifi-csi/
        │   │   │   │   ├── frame001.mat
        │   │   │   │   ├── frame002.mat
        │   │   │   │   └── ...
        │   │   │   └── ground_truth.npy

    Each CSI .mat file is expected to contain:

        CSIamp   shape: [3, 114, 10]
        CSIphase shape: [3, 114, 10]

    Assumption:

        frame001.mat -> ground_truth[0]
        frame002.mat -> ground_truth[1]
        frame003.mat -> ground_truth[2]

    Returned CSI layout by default:

        x shape: [T, A, S, C]

    where:

        T = temporal window, usually 10
        A = receive antennas, usually 3
        S = subcarriers, usually 114
        C = channels
    """

    VALID_RETURN_TYPES = {
        "amp",
        "phase",
        "amp_phase",
        "complex",
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: Optional[Dict[str, List[str]]] = None,
        return_type: str = "amp_phase",
        layout: str = "TASC",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        normalize: bool = False,
        include_metadata: bool = True,
        require_pose: bool = True,
    ):
        self.root = Path(root)
        self.split = split or {}
        self.return_type = return_type
        self.layout = layout
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.normalize = normalize
        self.include_metadata = include_metadata
        self.require_pose = require_pose

        self.pose_cache = {}

        if self.return_type not in self.VALID_RETURN_TYPES:
            raise ValueError(
                f"Invalid return_type={return_type}. "
                f"Expected one of {sorted(self.VALID_RETURN_TYPES)}."
            )

        if self.layout not in {"TASC", "CAST", "CTAS"}:
            raise ValueError(
                f"Invalid layout={layout}. Expected one of: 'TASC', 'CAST', 'CTAS'."
            )

        self.samples = self._index_dataset()

        self.environments = sorted({sample["environment"] for sample in self.samples})
        self.subjects = sorted({sample["subject"] for sample in self.samples})
        self.actions = sorted({sample["action"] for sample in self.samples})

        self.env_to_idx = {name: idx for idx, name in enumerate(self.environments)}
        self.subject_to_idx = {name: idx for idx, name in enumerate(self.subjects)}
        self.action_to_idx = {name: idx for idx, name in enumerate(self.actions)}

    def _index_dataset(self) -> List[Dict]:
        samples = []

        allowed_envs = set(self.split.get("environments", []))
        allowed_subjects = set(self.split.get("subjects", []))
        allowed_actions = set(self.split.get("actions", []))

        for env_dir in sorted(self.root.glob("E*")):
            if not env_dir.is_dir():
                continue

            environment = env_dir.name
            if allowed_envs and environment not in allowed_envs:
                continue

            for subject_dir in sorted(env_dir.glob("S*")):
                if not subject_dir.is_dir():
                    continue

                subject = subject_dir.name
                if allowed_subjects and subject not in allowed_subjects:
                    continue

                for action_dir in sorted(subject_dir.glob("A*")):
                    if not action_dir.is_dir():
                        continue

                    action = action_dir.name
                    if allowed_actions and action not in allowed_actions:
                        continue

                    wifi_dir = action_dir / "wifi-csi"
                    pose_path = action_dir / "ground_truth.npy"

                    if not wifi_dir.exists():
                        continue

                    if self.require_pose and not pose_path.exists():
                        raise FileNotFoundError(
                            f"Missing ground_truth.npy in: {action_dir}"
                        )

                    if not pose_path.exists():
                        pose_path = None

                    for mat_path in sorted(wifi_dir.glob("*.mat")):
                        samples.append(
                            {
                                "path": mat_path,
                                "pose_path": pose_path,
                                "environment": environment,
                                "subject": subject,
                                "action": action,
                                "frame": mat_path.stem,
                            }
                        )

        if len(samples) == 0:
            raise RuntimeError(
                f"No MM-Fi WiFi CSI .mat files found under: {self.root}"
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        amp, phase = self._load_csi_mat(sample["path"])
        x = self._build_input_tensor(amp, phase)

        if self.normalize:
            x = self._normalize(x)

        x = torch.as_tensor(x, dtype=self.dtype)

        target = {
            "environment": self.env_to_idx[sample["environment"]],
            "subject": self.subject_to_idx[sample["subject"]],
            "action": self.action_to_idx[sample["action"]],
        }

        pose_index = None

        if sample["pose_path"] is not None:
            poses = self._load_pose_npy(sample["pose_path"])
            pose_index = self._frame_to_index(sample["frame"])

            if pose_index < 0 or pose_index >= len(poses):
                raise IndexError(
                    f"Pose index {pose_index} is out of range for "
                    f"{sample['pose_path']} with length {len(poses)}"
                )

            pose = poses[pose_index]
            pose = torch.as_tensor(pose, dtype=self.dtype)

            target["pose"] = pose

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            x = self.transform(x)

        if self.include_metadata:
            metadata = {
                "path": str(sample["path"]),
                "pose_path": str(sample["pose_path"])
                if sample["pose_path"] is not None
                else None,
                "pose_index": pose_index,
                "environment_name": sample["environment"],
                "subject_name": sample["subject"],
                "action_name": sample["action"],
                "frame": sample["frame"],
            }

            return {
                "x": x,
                "target": target,
                "metadata": metadata,
            }

        return x, target

    def _load_csi_mat(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        data = loadmat(path)

        if "CSIamp" not in data:
            raise KeyError(f"Missing key 'CSIamp' in file: {path}")

        if "CSIphase" not in data:
            raise KeyError(f"Missing key 'CSIphase' in file: {path}")

        amp = data["CSIamp"].astype(np.float32)
        phase = data["CSIphase"].astype(np.float32)

        expected_shape = (3, 114, 10)

        if amp.shape != expected_shape:
            raise ValueError(
                f"Unexpected CSIamp shape in {path}: {amp.shape}. "
                f"Expected {expected_shape}."
            )

        if phase.shape != expected_shape:
            raise ValueError(
                f"Unexpected CSIphase shape in {path}: {phase.shape}. "
                f"Expected {expected_shape}."
            )

        return amp, phase

    def _load_pose_npy(self, path: Path) -> np.ndarray:
        if path not in self.pose_cache:
            self.pose_cache[path] = np.load(path, allow_pickle=True)

        return self.pose_cache[path]

    def _frame_to_index(self, frame_name: str) -> int:
        match = re.search(r"\d+", frame_name)

        if match is None:
            raise ValueError(f"Could not extract frame number from: {frame_name}")

        return int(match.group()) - 1

    def _build_input_tensor(self, amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Input amp/phase shape:
            [A, S, T]

        Internal canonical shape before layout conversion:
            [T, A, S, C]
        """

        amp = np.transpose(amp, (2, 0, 1))
        phase = np.transpose(phase, (2, 0, 1))

        if self.return_type == "amp":
            x = amp[..., None]

        elif self.return_type == "phase":
            x = phase[..., None]

        elif self.return_type == "amp_phase":
            x = np.stack([amp, phase], axis=-1)

        elif self.return_type == "complex":
            real = amp * np.cos(phase)
            imag = amp * np.sin(phase)
            x = np.stack([real, imag], axis=-1)

        else:
            raise RuntimeError(f"Unhandled return_type: {self.return_type}")

        if self.layout == "TASC":
            return x

        if self.layout == "CAST":
            return np.transpose(x, (3, 1, 2, 0))

        if self.layout == "CTAS":
            return np.transpose(x, (3, 0, 1, 2))

        raise RuntimeError(f"Unhandled layout: {self.layout}")

    @staticmethod
    def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + eps)


if __name__ == "__main__":
    dataset = MMFiWiFiCSIDataset(
        root=r"/home/buibaongan/PROJECT/DEEP LEARNING PROJECT/MMFi_Dataset/MMFi_Dataset",
        return_type="amp_phase",
        layout="CTAS",
        normalize=True,
        require_pose=True,
    )

    print("Number of samples:", len(dataset))

    item = dataset[0]

    print("x shape:", item["x"].shape)
    print("pose shape:", item["target"]["pose"].shape)
    print("target keys:", item["target"].keys())
    print("metadata:", item["metadata"])