# wimans_loader.py

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .wimans_reader import iter_wimans_samples, load_wimans_amp
except ImportError:  # pragma: no cover
    from wimans_reader import iter_wimans_samples, load_wimans_amp


class WiMANSDataset(Dataset):
    """
    PyTorch Dataset for WiMANS.

    Expected structure:

        root/
        ├── annotation.csv
        ├── wifi_csi/
        │   ├── amp/
        │   │   ├── act_1_1.npy
        │   │   └── ...
        │   └── mat/
        │       ├── act_1_1.mat
        │       └── ...
        └── video/
            ├── act_1_1.mp4
            └── ...

    annotation.csv contains:

        label
        environment
        wifi_band
        number_of_users
        user_1_location ... user_6_location
        user_1_activity ... user_6_activity

    WiMANS does not contain pose coordinates.
    """

    VALID_LAYOUTS = {
        "CTARS",  # [channel, time, tx, rx, subcarrier]
        "TARSC",  # [time, tx, rx, subcarrier, channel]
        "TRSC",   # [time, tx_rx, subcarrier, channel]
    }

    def __init__(
        self,
        root: Union[str, Path],
        layout: str = "CTARS",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        normalize: bool = False,
        single_person_only: bool = True,
        include_empty_room: bool = False,
        include_metadata: bool = True,
        max_packets: Optional[int] = None,
        pad_to_max_packets: bool = False,
    ):
        self.root = Path(root)
        self.layout = layout
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.normalize = normalize
        self.single_person_only = single_person_only
        self.include_empty_room = include_empty_room
        self.include_metadata = include_metadata
        self.max_packets = max_packets
        self.pad_to_max_packets = pad_to_max_packets

        if self.layout not in self.VALID_LAYOUTS:
            raise ValueError(
                f"Invalid layout={layout}. Expected one of {sorted(self.VALID_LAYOUTS)}."
            )

        if self.pad_to_max_packets and self.max_packets is None:
            raise ValueError("pad_to_max_packets=True requires max_packets to be set.")

        self.samples = self._index_dataset()

        self.environments = sorted({s["environment"] for s in self.samples})
        self.wifi_bands = sorted({s["wifi_band"] for s in self.samples})
        self.activities = sorted(
            {
                activity
                for s in self.samples
                for activity in s["activities"]
                if activity
            }
        )
        self.locations = sorted(
            {
                location
                for s in self.samples
                for location in s["locations"]
                if location
            }
        )

        self.environment_to_idx = {
            name: idx for idx, name in enumerate(self.environments)
        }
        self.wifi_band_to_idx = {
            name: idx for idx, name in enumerate(self.wifi_bands)
        }
        self.activity_to_idx = {
            name: idx for idx, name in enumerate(self.activities)
        }
        self.location_to_idx = {
            name: idx for idx, name in enumerate(self.locations)
        }

    def _index_dataset(self) -> List[Dict]:
        return iter_wimans_samples(
            self.root,
            single_person_only=self.single_person_only,
            include_empty_room=self.include_empty_room,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        amp = load_wimans_amp(sample["amp_path"])
        x = self._build_input_tensor(amp)

        if self.max_packets is not None:
            x = self._fix_packet_length(x)

        if self.normalize:
            x = self._normalize(x)

        x = torch.as_tensor(x, dtype=self.dtype)

        activity_names = sample["activities"]
        location_names = sample["locations"]

        target = {
            "num_users": sample["num_users"],
            "environment": self.environment_to_idx[sample["environment"]],
            "wifi_band": self.wifi_band_to_idx[sample["wifi_band"]],
            "activities": [
                self.activity_to_idx[name] for name in activity_names
            ],
            "locations": [
                self.location_to_idx[name] for name in location_names
            ],
        }

        if self.single_person_only:
            target["activity"] = target["activities"][0]
            target["location"] = target["locations"][0]

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            x = self.transform(x)

        if self.include_metadata:
            metadata = {
                "label": sample["label"],
                "amp_path": str(sample["amp_path"]),
                "mat_path": str(sample["mat_path"]) if sample["mat_path"] else None,
                "video_path": str(sample["video_path"]) if sample["video_path"] else None,
                "environment_name": sample["environment"],
                "wifi_band_name": sample["wifi_band"],
                "num_users": sample["num_users"],
                "activity_names": activity_names,
                "location_names": location_names,
            }

            return {
                "x": x,
                "target": target,
                "metadata": metadata,
            }

        return x, target

    def _build_input_tensor(self, amp: np.ndarray) -> np.ndarray:
        """
        Input amp shape from your sample:

            [time, tx, rx, subcarrier]

        Example:

            [2901, 3, 3, 30]

        This is amplitude-only CSI, so channel = 1.
        """

        amp = np.asarray(amp, dtype=np.float32)

        if amp.ndim != 4:
            raise ValueError(
                f"Unexpected WiMANS amp shape: {amp.shape}. "
                f"Expected [time, tx, rx, subcarrier]."
            )

        if amp.shape[1:] != (3, 3, 30):
            raise ValueError(
                f"Unexpected WiMANS amp shape: {amp.shape}. "
                f"Expected [time, 3, 3, 30]."
            )

        x = amp[..., None]

        if self.layout == "TARSC":
            return x

        if self.layout == "CTARS":
            return np.transpose(x, (4, 0, 1, 2, 3))

        if self.layout == "TRSC":
            time, tx, rx, subcarrier, channel = x.shape
            return x.reshape(time, tx * rx, subcarrier, channel)

        raise RuntimeError(f"Unhandled layout: {self.layout}")

    def _fix_packet_length(self, x: np.ndarray) -> np.ndarray:
        if self.layout == "CTARS":
            time_dim = 1
        else:
            time_dim = 0

        num_packets = x.shape[time_dim]

        if num_packets > self.max_packets:
            index = [slice(None)] * x.ndim
            index[time_dim] = slice(0, self.max_packets)
            return x[tuple(index)]

        if num_packets < self.max_packets and self.pad_to_max_packets:
            pad_width = [(0, 0)] * x.ndim
            pad_width[time_dim] = (0, self.max_packets - num_packets)
            return np.pad(x, pad_width, mode="constant", constant_values=0)

        return x

    @staticmethod
    def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + eps)


if __name__ == "__main__":
    dataset = WiMANSDataset(
        root=r"/home/buibaongan/rf-worldpose/data/bronze/WiMANS",
        layout="CTARS",
        normalize=True,
        single_person_only=True,
        include_empty_room=False,
        max_packets=3000,
        pad_to_max_packets=True,
    )

    print("Number of samples:", len(dataset))

    item = dataset[0]

    print("x shape:", item["x"].shape)
    print("target:", item["target"])
    print("metadata:", item["metadata"])
