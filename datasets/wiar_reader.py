from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import re

try:
    from .read_csi import ACTIVITY_LABELS, load_wiar_file
except ImportError:  # pragma: no cover
    from read_csi import ACTIVITY_LABELS, load_wiar_file


def parse_wiar_filename(path: Union[str, Path]) -> Optional[Tuple[int, int]]:
    path = Path(path)
    match = re.match(r"csi_a(\d+)_(\d+)\.dat$", path.name)

    if match is None:
        return None

    return int(match.group(1)), int(match.group(2))


def get_wiar_volunteer_name(path: Union[str, Path]) -> str:
    path = Path(path)

    for part in reversed(path.parts):
        if part.lower().startswith("volunteer"):
            return part

    return "unknown"


def index_wiar_samples(
    root: Union[str, Path],
    *,
    split: Optional[Dict[str, List]] = None,
) -> List[Dict]:
    root = Path(root)
    split = split or {}
    samples = []

    allowed_volunteers = set(split.get("volunteers", []))
    allowed_actions = set(split.get("actions", []))

    for path in sorted(root.rglob("*.dat")):
        parsed = parse_wiar_filename(path)

        if parsed is None:
            continue

        activity_id, sample_id = parsed
        volunteer = get_wiar_volunteer_name(path)

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
        raise RuntimeError(f"No WiAR .dat files found under: {root}")

    return samples


def load_wiar_sample(path: Union[str, Path]) -> dict:
    return load_wiar_file(str(path))
