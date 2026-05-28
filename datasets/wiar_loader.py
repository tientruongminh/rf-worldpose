# wiar_loader.py

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import sys

import numpy as np
import torch
from torch.utils.data import Dataset


# If read_csi.py is in another folder, add that folder here.
READ_CSI_DIR = r"C:\Users\Ngan\Downloads\DEEP LEARNING PROJECT"

if READ_CSI_DIR not in sys.path:
    sys.path.append(READ_CSI_DIR)

from read_csi import load_wiar_file, ACTIVITY_LABELS


class WiARDataset(Dataset):
    """
    PyTorch Dataset for WiAR.

    Expected structure:

        root/
        ├── volunteer_7/
        │   └── volunteer_7/
        │       ├── csi_a1_1.dat
        │       ├── csi_a1_2.dat
        │       └── ...
        ├── volunteer_8/
        │   └── volunteer_8/
        │       └── ...
        └── ...

    Filename format:

        csi_a{activity_id}_{sample_id}.dat

    Example:

        csi_a5_2.dat

    means:

        activity_id = 5
        activity    = draw_x
        sample_id   = 2

    WiAR has action/activity labels only.
    It does NOT contain pose labels.
    """

    VALID_RETURN_TYPES = {
        "amp",
        "phase",
        "amp_phase",
        "complex",
    }

    VALID_LAYOUTS = {
        "CTARS",  # [channel, time, tx, rx, subcarrier]
        "TARSC",  # [time, tx, rx, subcarrier, channel]
        "TRSC",   # [time, tx_rx, subcarrier, channel]
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: Optional[Dict[str, List]] = None,
        return_type: str = "amp_phase",
        layout: str = "CTARS",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        normalize: bool = False,
        include_metadata: bool = True,
        max_packets: Optional[int] = None,
        pad_to_max_packets: bool = False,
    ):
        """
        Args:
            root:
                Path to WiAR data root.

                Example:
                    C:/Users/Ngan/Downloads/DEEP LEARNING PROJECT/WiAR-master/WiAR-master/data/data

            split:
                Optional filtering dictionary.

                Example:
                    split = {
                        "volunteers": ["volunteer_7", "volunteer_8"],
                        "actions": [1, 2, 3],
                    }

            return_type:
                "amp"       -> amplitude only
                "phase"     -> phase only
                "amp_phase" -> amplitude + phase
                "complex"   -> real + imaginary

            layout:
                "CTARS" -> [channel, time, tx, rx, subcarrier]
                "TARSC" -> [time, tx, rx, subcarrier, channel]
                "TRSC"  -> [time, tx_rx, subcarrier, channel]

            max_packets:
                If not None, truncates CSI to this many packets.

            pad_to_max_packets:
                If True, pads shorter samples to max_packets.
                Useful because WiAR files can have different packet counts.
        """

        self.root = Path(root)
        self.split = split or {}
        self.return_type = return_type
        self.layout = layout
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.normalize = normalize
        self.include_metadata = include_metadata
        self.max_packets = max_packets
        self.pad_to_max_packets = pad_to_max_packets

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

        if self.pad_to_max_packets and self.max_packets is None:
            raise ValueError("pad_to_max_packets=True requires max_packets to be set.")

        self.samples = self._index_dataset()

        self.volunteers = sorted({sample["volunteer"] for sample in self.samples})
        self.actions = sorted({sample["activity_id"] for sample in self.samples})

        self.volunteer_to_idx = {
            name: idx for idx, name in enumerate(self.volunteers)
        }

    def _index_dataset(self) -> List[Dict]:
        samples = []

        allowed_volunteers = set(self.split.get("volunteers", []))
        allowed_actions = set(self.split.get("actions", []))

        for path in sorted(self.root.rglob("*.dat")):
            parsed = self._parse_wiar_filename(path)

            if parsed is None:
                continue

            activity_id, sample_id = parsed
            volunteer = self._get_volunteer_name(path)

            if allowed_volunteers and volunteer not in allowed_volunteers:
                continue

            if allowed_actions and activity_id not in allowed_actions:
                continue

            samples.append(
                {
                    "path": path,
                    "volunteer": volunteer,
                    "activity_id": activity_id,
                    "activity_name": ACTIVITY_LABELS.get(
                        activity_id, f"unknown_activity_{activity_id}"
                    ),
                    "sample_id": sample_id,
                }
            )

        if len(samples) == 0:
            raise RuntimeError(f"No WiAR .dat files found under: {self.root}")

        return samples

    @staticmethod
    def _parse_wiar_filename(path: Path):
        """
        csi_a5_2.dat -> activity_id=5, sample_id=2
        """

        import re

        match = re.match(r"csi_a(\d+)_(\d+)\.dat$", path.name)

        if match is None:
            return None

        activity_id = int(match.group(1))
        sample_id = int(match.group(2))

        return activity_id, sample_id

    @staticmethod
    def _get_volunteer_name(path: Path) -> str:
        for part in reversed(path.parts):
            if part.lower().startswith("volunteer"):
                return part

        return "unknown"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_info = self.samples[index]

        loaded = load_wiar_file(str(sample_info["path"]))

        csi = loaded["csi"]
        x = self._build_input_tensor(csi)

        if self.max_packets is not None:
            x = self._fix_packet_length(x)

        if self.normalize:
            x = self._normalize(x)

        x = torch.as_tensor(x, dtype=self.dtype)

        target = {
            "activity": sample_info["activity_id"] - 1,
            "activity_id": sample_info["activity_id"],
            "volunteer": self.volunteer_to_idx[sample_info["volunteer"]],
            "sample_id": sample_info["sample_id"],
        }

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            x = self.transform(x)

        if self.include_metadata:
            metadata = {
                "path": str(sample_info["path"]),
                "filename": sample_info["path"].name,
                "volunteer_name": sample_info["volunteer"],
                "activity_name": sample_info["activity_name"],
                "activity_id": sample_info["activity_id"],
                "sample_id": sample_info["sample_id"],
                "num_packets": loaded["num_packets"],
                "raw_csi_shape": loaded["csi"].shape,
            }

            return {
                "x": x,
                "target": target,
                "metadata": metadata,
            }

        return x, target

    def _build_input_tensor(self, csi: np.ndarray) -> np.ndarray:
        """
        read_csi.py returns CSI as:

            [time, subcarrier, rx, tx]

        Convert to canonical:

            [time, tx, rx, subcarrier, channel]
        """

        csi = np.asarray(csi)

        if csi.ndim != 4:
            raise ValueError(
                f"Unexpected CSI shape: {csi.shape}. "
                f"Expected [time, subcarrier, rx, tx]."
            )

        # [time, subcarrier, rx, tx] -> [time, tx, rx, subcarrier]
        csi = np.transpose(csi, (0, 3, 2, 1))

        if self.return_type == "amp":
            x = np.abs(csi).astype(np.float32)[..., None]

        elif self.return_type == "phase":
            x = np.angle(csi).astype(np.float32)[..., None]

        elif self.return_type == "amp_phase":
            amp = np.abs(csi).astype(np.float32)
            phase = np.angle(csi).astype(np.float32)
            x = np.stack([amp, phase], axis=-1)

        elif self.return_type == "complex":
            real = np.real(csi).astype(np.float32)
            imag = np.imag(csi).astype(np.float32)
            x = np.stack([real, imag], axis=-1)

        else:
            raise RuntimeError(f"Unhandled return_type: {self.return_type}")

        if self.layout == "TARSC":
            return x

        if self.layout == "CTARS":
            return np.transpose(x, (4, 0, 1, 2, 3))

        if self.layout == "TRSC":
            time, tx, rx, subcarrier, channel = x.shape
            return x.reshape(time, tx * rx, subcarrier, channel)

        raise RuntimeError(f"Unhandled layout: {self.layout}")

    def _fix_packet_length(self, x: np.ndarray) -> np.ndarray:
        """
        Truncate or pad the time dimension.

        For CTARS:
            [channel, time, tx, rx, subcarrier]

        For TARSC:
            [time, tx, rx, subcarrier, channel]

        For TRSC:
            [time, tx_rx, subcarrier, channel]
        """

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
    dataset = WiARDataset(
        root=r"/home/buibaongan/PROJECT/DEEP LEARNING PROJECT/WiAR-master/WiAR-master/data/data",
        return_type="amp_phase",
        layout="CTARS",
        normalize=True,
        max_packets=300,
        pad_to_max_packets=True,
    )

    print("Number of samples:", len(dataset))

    item = dataset[0]

    print("x shape:", item["x"].shape)
    print("target:", item["target"])
    print("metadata:", item["metadata"])