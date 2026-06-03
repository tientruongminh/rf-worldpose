# person_in_wifi_3d_loader.py

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .person_wifi_reader import (
        get_person_wifi_split_dir,
        index_person_wifi_samples,
        load_person_wifi_csi_mat,
        load_person_wifi_keypoint_npy,
    )
except ImportError:  # pragma: no cover
    from person_wifi_reader import (
        get_person_wifi_split_dir,
        index_person_wifi_samples,
        load_person_wifi_csi_mat,
        load_person_wifi_keypoint_npy,
    )


class PersonInWiFi3DDataset(Dataset):
    """
    PyTorch Dataset for Person-in-WiFi 3D.

    Expected structure:

        root/
        ├── train_data/
        │   ├── train_data_list.txt
        │   ├── csi/
        │   │   ├── S11_01_10.mat
        │   │   └── ...
        │   └── keypoint/
        │       ├── S11_01_10.npy
        │       └── ...
        └── test_data/
            ├── test_data_list.txt
            ├── csi/
            └── keypoint/

    Each sample has:

        CSI:
            .mat file containing key "csi_out"

        Pose:
            .npy file with shape [num_people, 14, 3]

    This loader keeps only single-person samples by default:

        keypoint.shape[0] == 1

    Returned x layout by default:

        x shape: [2, 20, 3, 3, 30]

    where:

        2  = amplitude + phase
        20 = time packets
        3  = antenna dimension
        3  = antenna dimension
        30 = subcarriers

    Returned pose shape:

        pose shape: [14, 3]
    """

    VALID_RETURN_TYPES = {
        "amp",
        "phase",
        "amp_phase",
        "complex",
    }

    VALID_LAYOUTS = {
        "CTARS",  # [channel, time, ant1, ant2, subcarrier]
        "TARSC",  # [time, ant1, ant2, subcarrier, channel]
        "ATSC",   # [ant_pair, time, subcarrier, channel], antenna pair flattened
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        return_type: str = "amp_phase",
        layout: str = "CTARS",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        normalize: bool = False,
        single_person_only: bool = True,
        squeeze_person_dim: bool = True,
        include_metadata: bool = True,
    ):
        self.root = Path(root)
        self.split = split.lower()
        self.return_type = return_type
        self.layout = layout
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.normalize = normalize
        self.single_person_only = single_person_only
        self.squeeze_person_dim = squeeze_person_dim
        self.include_metadata = include_metadata

        if self.return_type not in self.VALID_RETURN_TYPES:
            raise ValueError(
                f"Invalid return_type={return_type}. "
                f"Expected one of {sorted(self.VALID_RETURN_TYPES)}."
            )

        if self.layout not in self.VALID_LAYOUTS:
            raise ValueError(
                f"Invalid layout={layout}. "
                f"Expected one of {sorted(self.VALID_LAYOUTS)}."
            )

        self.split_dir = self._get_split_dir()
        self.samples = self._index_dataset()

    def _get_split_dir(self) -> Path:
        return get_person_wifi_split_dir(self.root, self.split)

    def _index_dataset(self):
        return index_person_wifi_samples(
            self.root,
            split=self.split,
            single_person_only=self.single_person_only,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        csi = load_person_wifi_csi_mat(sample["csi_path"])
        keypoint = load_person_wifi_keypoint_npy(
            sample["keypoint_path"],
            single_person_only=self.single_person_only,
        )

        x = self._build_input_tensor(csi)

        if self.normalize:
            x = self._normalize(x)

        if self.single_person_only and self.squeeze_person_dim:
            keypoint = keypoint[0]

        x = torch.as_tensor(x, dtype=self.dtype)
        pose = torch.as_tensor(keypoint, dtype=self.dtype)

        target = {
            "pose": pose,
            "num_people": sample["num_people"],
        }

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            x = self.transform(x)

        if self.include_metadata:
            metadata = {
                "name": sample["name"],
                "split": sample["split"],
                "csi_path": str(sample["csi_path"]),
                "keypoint_path": str(sample["keypoint_path"]),
                "num_people": sample["num_people"],
            }

            return {
                "x": x,
                "target": target,
                "metadata": metadata,
            }

        return x, target

    def _build_input_tensor(self, csi: np.ndarray) -> np.ndarray:
        """
        Input csi shape:
            [ant1, ant2, subcarrier, time]

        Canonical layout:
            [time, ant1, ant2, subcarrier, channel]
        """

        amp = np.abs(csi).astype(np.float32)
        phase = np.angle(csi).astype(np.float32)

        amp = np.transpose(amp, (3, 0, 1, 2))
        phase = np.transpose(phase, (3, 0, 1, 2))

        if self.return_type == "amp":
            x = amp[..., None]

        elif self.return_type == "phase":
            x = phase[..., None]

        elif self.return_type == "amp_phase":
            x = np.stack([amp, phase], axis=-1)

        elif self.return_type == "complex":
            real = np.real(csi).astype(np.float32)
            imag = np.imag(csi).astype(np.float32)

            real = np.transpose(real, (3, 0, 1, 2))
            imag = np.transpose(imag, (3, 0, 1, 2))

            x = np.stack([real, imag], axis=-1)

        else:
            raise RuntimeError(f"Unhandled return_type: {self.return_type}")

        if self.layout == "TARSC":
            return x

        if self.layout == "CTARS":
            return np.transpose(x, (4, 0, 1, 2, 3))

        if self.layout == "ATSC":
            time, ant1, ant2, subcarrier, channel = x.shape
            return x.reshape(time, ant1 * ant2, subcarrier, channel)

        raise RuntimeError(f"Unhandled layout: {self.layout}")

    @staticmethod
    def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + eps)


if __name__ == "__main__":
    dataset = PersonInWiFi3DDataset(
        root=r"/home/buibaongan/rf-worldpose/data/bronze/wifipose_dataset",
        split="train",
        return_type="amp_phase",
        layout="CTARS",
        normalize=True,
        single_person_only=True,
    )

    print("Number of single-person samples:", len(dataset))

    item = dataset[0]

    print("x shape:", item["x"].shape)
    print("pose shape:", item["target"]["pose"].shape)
    print("num_people:", item["target"]["num_people"])
    print("metadata:", item["metadata"])
