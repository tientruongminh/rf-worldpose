from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np


def get_person_wifi_split_dir(root: Union[str, Path], split: str) -> Path:
    root = Path(root)
    split = split.lower()

    if split in {"train", "training"}:
        return root / "train_data"

    if split in {"test", "testing"}:
        return root / "test_data"

    raise ValueError("split must be 'train' or 'test'.")


def load_person_wifi_name_list(path: Union[str, Path]) -> List[str]:
    names = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()

            if name:
                names.append(name.split()[0])

    return names


def get_person_wifi_num_people(keypoint_path: Union[str, Path]) -> int:
    keypoint = np.load(keypoint_path, allow_pickle=True)

    if keypoint.ndim == 3:
        return keypoint.shape[0]

    if keypoint.ndim == 2:
        return 1

    raise ValueError(
        f"Unexpected keypoint shape in {keypoint_path}: {keypoint.shape}. "
        f"Expected [num_people, 14, 3] or [14, 3]."
    )


def index_person_wifi_samples(
    root: Union[str, Path],
    *,
    split: str = "train",
    single_person_only: bool = True,
) -> List[Dict]:
    split = split.lower()
    split_dir = get_person_wifi_split_dir(root, split)
    list_path = split_dir / f"{split}_data_list.txt"

    if not list_path.exists():
        raise FileNotFoundError(f"Missing data list file: {list_path}")

    names = load_person_wifi_name_list(list_path)
    samples = []

    for name in names:
        csi_path = split_dir / "csi" / f"{name}.mat"
        keypoint_path = split_dir / "keypoint" / f"{name}.npy"

        if not csi_path.exists():
            raise FileNotFoundError(f"Missing CSI file: {csi_path}")

        if not keypoint_path.exists():
            raise FileNotFoundError(f"Missing keypoint file: {keypoint_path}")

        num_people = get_person_wifi_num_people(keypoint_path)

        if single_person_only and num_people != 1:
            continue

        samples.append(
            {
                "name": name,
                "csi_path": csi_path,
                "keypoint_path": keypoint_path,
                "num_people": num_people,
                "split": split,
            }
        )

    if len(samples) == 0:
        raise RuntimeError(
            f"No samples found for split={split}. "
            f"If single_person_only=True, no single-person samples were found."
        )

    return samples


def load_person_wifi_csi_mat(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)

    with h5py.File(path, "r") as data:
        if "csi_out" not in data:
            raise KeyError(
                f"Missing key 'csi_out' in {path}. "
                f"Available keys: {list(data.keys())}"
            )

        raw = np.array(data["csi_out"])

    if raw.dtype.fields is not None:
        if "real" not in raw.dtype.fields or "imag" not in raw.dtype.fields:
            raise ValueError(
                f"Structured CSI in {path} does not contain real/imag fields. "
                f"Fields: {raw.dtype.fields.keys()}"
            )

        csi = raw["real"] + raw["imag"] * 1j
    else:
        csi = raw

    if csi.shape != (3, 3, 30, 20) and csi.ndim == 4:
        csi = np.transpose(csi, (3, 2, 1, 0))

    expected_shape = (3, 3, 30, 20)

    if csi.shape != expected_shape:
        raise ValueError(
            f"Unexpected CSI shape in {path}: {csi.shape}. "
            f"Expected {expected_shape} after loading."
        )

    return csi.astype(np.complex64)


def load_person_wifi_keypoint_npy(
    path: Union[str, Path],
    *,
    single_person_only: bool = True,
) -> np.ndarray:
    keypoint = np.load(path, allow_pickle=True)
    keypoint = np.asarray(keypoint, dtype=np.float32)

    if keypoint.ndim == 2:
        keypoint = keypoint[None, ...]

    expected_joint_shape = (14, 3)

    if keypoint.ndim != 3 or keypoint.shape[1:] != expected_joint_shape:
        raise ValueError(
            f"Unexpected keypoint shape in {path}: {keypoint.shape}. "
            f"Expected [num_people, 14, 3]."
        )

    if single_person_only and keypoint.shape[0] != 1:
        raise ValueError(
            f"Expected single-person keypoint in {path}, "
            f"but found shape {keypoint.shape}."
        )

    return keypoint
