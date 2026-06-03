from pathlib import Path
from typing import Dict, List, Union

import csv
import numpy as np


def iter_wimans_samples(
    root: Union[str, Path],
    *,
    single_person_only: bool = True,
    include_empty_room: bool = False,
) -> List[Dict]:
    root = Path(root)
    annotation_path = root / "annotation.csv"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    samples = []

    with open(annotation_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label = row["label"].strip()
            num_users = int(row["number_of_users"])

            if single_person_only and num_users != 1:
                continue

            if not include_empty_room and num_users == 0:
                continue

            amp_path = root / "wifi_csi" / "amp" / f"{label}.npy"
            mat_path = root / "wifi_csi" / "mat" / f"{label}.mat"
            video_path = root / "video" / f"{label}.mp4"

            if not amp_path.exists():
                continue

            locations = []
            activities = []

            for user_idx in range(1, 7):
                location = row.get(f"user_{user_idx}_location", "").strip()
                activity = row.get(f"user_{user_idx}_activity", "").strip()

                if location:
                    locations.append(location)

                if activity:
                    activities.append(activity)

            samples.append(
                {
                    "label": label,
                    "amp_path": amp_path,
                    "mat_path": mat_path if mat_path.exists() else None,
                    "video_path": video_path if video_path.exists() else None,
                    "environment": row["environment"].strip(),
                    "wifi_band": row["wifi_band"].strip(),
                    "num_users": num_users,
                    "locations": locations,
                    "activities": activities,
                }
            )

    if len(samples) == 0:
        raise RuntimeError(
            f"No WiMANS samples found under {root}. "
            f"single_person_only={single_person_only}"
        )

    return samples


def load_wimans_amp(path: Union[str, Path]) -> np.ndarray:
    return np.load(path, allow_pickle=False)
