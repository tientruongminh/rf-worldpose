from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import re

import numpy as np
from scipy.io import loadmat


def index_mmfi_wifi_csi_samples(
    root: Union[str, Path],
    *,
    split: Optional[Dict[str, List[str]]] = None,
    require_pose: bool = True,
) -> List[Dict]:
    root = Path(root)
    split = split or {}
    samples = []

    allowed_envs = set(split.get("environments", []))
    allowed_subjects = set(split.get("subjects", []))
    allowed_actions = set(split.get("actions", []))

    for env_dir in sorted(root.glob("E*")):
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

                if require_pose and not pose_path.exists():
                    raise FileNotFoundError(f"Missing ground_truth.npy in: {action_dir}")

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
        raise RuntimeError(f"No MM-Fi WiFi CSI .mat files found under: {root}")

    return samples


def load_mmfi_csi_mat(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
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


def load_mmfi_pose_npy(path: Union[str, Path]) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def mmfi_frame_to_index(frame_name: str) -> int:
    match = re.search(r"\d+", frame_name)

    if match is None:
        raise ValueError(f"Could not extract frame number from: {frame_name}")

    return int(match.group()) - 1
