from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import re

import h5py
import numpy as np


def get_wipose_split_name(path: Path) -> str:
    parent = path.parent.name

    if parent.lower() in {"train", "test"}:
        return parent

    return "unknown"


def get_wipose_action_name(path: Path) -> str:
    stem = path.stem
    match = re.match(r"(.+)_\d+-frame\d+", stem)

    if match is None:
        return stem.split("_")[0].lower()

    return match.group(1).lower()


def get_wipose_subject_name(path: Path) -> Optional[str]:
    match = re.search(r"_(\d+)-frame", path.stem)

    if match is None:
        return None

    return match.group(1)


def get_wipose_frame_name(path: Path) -> str:
    match = re.search(r"(frame\d+)", path.stem)

    if match is None:
        return path.stem

    return match.group(1)


def index_wipose_samples(root: Union[str, Path], split: Optional[str] = None) -> List[Dict]:
    root = Path(root)

    if split is None:
        files = sorted(root.rglob("*.mat"))
    else:
        files = sorted((root / split).glob("*.mat"))

    samples = [
        {
            "path": path,
            "split": get_wipose_split_name(path),
            "action": get_wipose_action_name(path),
            "subject": get_wipose_subject_name(path),
            "frame": get_wipose_frame_name(path),
        }
        for path in files
    ]

    if len(samples) == 0:
        raise RuntimeError(f"No Wi-Pose .mat files found under: {root}")

    return samples


def prepare_wipose_csi(csi: np.ndarray, path: Union[str, Path]) -> np.ndarray:
    csi = np.asarray(csi, dtype=np.float32)
    expected_shape = (3, 3, 30, 5)

    if csi.shape != expected_shape:
        raise ValueError(
            f"Unexpected CSI shape in {path}: {csi.shape}. "
            f"Expected {expected_shape}."
        )

    return csi[..., None]


def prepare_wipose_pose(pose: np.ndarray, path: Union[str, Path]) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).squeeze()

    if pose.shape == (54,):
        return pose.reshape(18, 3)

    if pose.shape == (1, 54):
        return pose.reshape(18, 3)

    if pose.shape == (18, 3):
        return pose

    if pose.shape == (3, 18):
        return pose.T

    raise ValueError(f"Unexpected SkeletonPoints shape in {path}: {pose.shape}")


def load_wipose_mat(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)

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

    return prepare_wipose_csi(csi, path), prepare_wipose_pose(pose, path)
