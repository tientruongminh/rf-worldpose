from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import logging
import time
import os
import sys
import tempfile

import numpy as np

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None

DAGSTER_ROOT = Path(__file__).resolve().parents[2]
if str(DAGSTER_ROOT) not in sys.path:
    sys.path.insert(0, str(DAGSTER_ROOT))

from rfpose_pipelines.etl.bronze_to_silver import (
    download_s3_file,
    is_s3_uri,
    upload_s3_directory,
    upload_s3_file,
)


log = logging.getLogger(__name__)

POSE_JOINTS = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def load_silver(path: str | Path) -> list[dict]:
    if is_s3_uri(path):
        with tempfile.TemporaryDirectory(prefix="rfpose-silver-in-") as tmpdir:
            local_path = Path(tmpdir) / Path(str(path)).name
            download_s3_file(path, local_path)
            return load_silver(local_path)

    path = Path(path)

    if path.suffix == ".parquet" and pl is not None:
        return pl.read_parquet(path).to_dicts()

    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_dataset_filter(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def stable_fraction(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def normalize_split(split: str | None, sample_key: str) -> str:
    if split is not None:
        split = str(split).lower()
        if split in {"train", "training"}:
            return "train"
        if split in {"val", "valid", "validation"}:
            return "val"
        if split in {"test", "testing"}:
            return "test"

    frac = stable_fraction(sample_key)
    if frac < 0.8:
        return "train"
    if frac < 0.9:
        return "val"
    return "test"


def label_index(label, label_map: dict[str, int]) -> int:
    if label is None:
        return -1
    label = str(label)
    if label not in label_map:
        label_map[label] = len(label_map)
    return label_map[label]


def pad_or_trim(values, n_subcarriers: int) -> np.ndarray:
    arr = np.asarray(values or [], dtype=np.float32)[:n_subcarriers]
    if len(arr) < n_subcarriers:
        arr = np.pad(arr, (0, n_subcarriers - len(arr)), mode="constant")
    return arr


def infer_n_subcarriers(rows: list[dict]) -> int:
    values = [int(row["n_subcarriers"]) for row in rows if row.get("n_subcarriers")]
    if values:
        return max(values)
    lengths = [len(row.get("amplitude") or []) for row in rows]
    if lengths:
        return max(lengths)
    raise ValueError("Cannot infer n_subcarriers from rows.")


def group_silver_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        if not row.get("dataset") or not row.get("sample_id"):
            continue
        if not row.get("amplitude"):
            continue
        grouped[(row["dataset"], str(row["sample_id"]))].append(row)
    return dict(grouped)


def sample_has_phase(rows: list[dict]) -> bool:
    return any(row.get("phase") for row in rows)


def build_antenna_pair_tensor(rows: list[dict]) -> np.ndarray:
    n_subcarriers = infer_n_subcarriers(rows)
    has_phase = sample_has_phase(rows)
    channels = 2 if has_phase else 1
    max_time = max(int(row["timestamp_us"]) for row in rows) + 1
    max_tx = max(int(row.get("tx") or 1) for row in rows)
    max_rx = max(int(row.get("rx") or 1) for row in rows)

    x = np.zeros((channels, max_time, max_tx, max_rx, n_subcarriers), dtype=np.float32)

    for row in rows:
        t = int(row["timestamp_us"])
        tx = int(row.get("tx") or 1) - 1
        rx = int(row.get("rx") or 1) - 1
        x[0, t, tx, rx, :] = pad_or_trim(row.get("amplitude"), n_subcarriers)
        if has_phase:
            x[1, t, tx, rx, :] = pad_or_trim(row.get("phase"), n_subcarriers)

    return x


def build_antenna_tensor(rows: list[dict]) -> np.ndarray:
    n_subcarriers = infer_n_subcarriers(rows)
    has_phase = sample_has_phase(rows)
    channels = 2 if has_phase else 1
    max_time = max(int(row["timestamp_us"]) for row in rows) + 1
    max_antenna = max(int(row.get("antenna") or row.get("rx") or row["node_id"]) for row in rows)

    x = np.zeros((channels, max_time, max_antenna, n_subcarriers), dtype=np.float32)

    for row in rows:
        t = int(row["timestamp_us"])
        antenna = int(row.get("antenna") or row.get("rx") or row["node_id"]) - 1
        x[0, t, antenna, :] = pad_or_trim(row.get("amplitude"), n_subcarriers)
        if has_phase:
            x[1, t, antenna, :] = pad_or_trim(row.get("phase"), n_subcarriers)

    return x


def build_csi_tensor(dataset: str, rows: list[dict]) -> np.ndarray:
    if dataset in {"wimans", "wipose", "person_in_wifi_3d", "wiar"}:
        return build_antenna_pair_tensor(rows)

    if dataset in {"mmfi", "uthar"}:
        return build_antenna_tensor(rows)

    return build_antenna_tensor(rows)


def get_pose(rows: list[dict]) -> tuple[np.ndarray, int]:
    for row in rows:
        if row.get("pose") is not None:
            pose = np.asarray(row["pose"], dtype=np.float32)
            if pose.shape != (13, 3):
                raise ValueError(f"Expected common pose shape [13, 3], got {pose.shape}")
            return pose, 1

    return np.zeros((13, 3), dtype=np.float32), 0


def make_windows(x: np.ndarray, window_frames: int, stride: int) -> list[tuple[int, np.ndarray]]:
    time_steps = x.shape[1]
    if time_steps < window_frames:
        return []

    windows = []
    for start in range(0, time_steps - window_frames + 1, stride):
        windows.append((start, x[:, start : start + window_frames, ...]))
    return windows


def build_gold_records(
    grouped: dict[tuple[str, str], list[dict]],
    *,
    window_frames: int,
    stride: int,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    label_maps = {
        "activity": {},
        "location": {},
        "environment": {},
        "subject": {},
    }
    records_by_dataset = defaultdict(list)

    for (dataset, sample_id), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (int(row["timestamp_us"]), int(row["node_id"])))
        first = rows[0]
        sample_key = f"{dataset}:{sample_id}"
        split = normalize_split(first.get("split"), sample_key)
        x = build_csi_tensor(dataset, rows)
        pose, pose_mask = get_pose(rows)

        activity_id = label_index(first.get("activity"), label_maps["activity"])
        location_id = label_index(first.get("location_key") or first.get("location"), label_maps["location"])
        environment_id = label_index(
            first.get("environment_key") or first.get("environment"),
            label_maps["environment"],
        )
        subject_id = label_index(first.get("subject_key") or first.get("subject"), label_maps["subject"])

        for window_start, window_x in make_windows(x, window_frames, stride):
            records_by_dataset[dataset].append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "window_start": window_start,
                    "x": window_x.astype(np.float32),
                    "pose": pose,
                    "pose_mask": pose_mask,
                    "activity_id": activity_id,
                    "activity_mask": int(activity_id >= 0),
                    "location_id": location_id,
                    "location_mask": int(location_id >= 0),
                    "environment_id": environment_id,
                    "environment_mask": int(environment_id >= 0),
                    "subject_id": subject_id,
                    "subject_mask": int(subject_id >= 0),
                }
            )

    return dict(records_by_dataset), label_maps


def one_hot(ids: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.zeros((len(ids), num_classes), dtype=np.float32)
    valid = ids >= 0
    if num_classes > 0 and valid.any():
        y[np.where(valid)[0], ids[valid]] = 1.0
    return y


def records_to_arrays(records: list[dict], label_maps: dict[str, dict[str, int]]) -> dict[str, np.ndarray]:
    x = np.stack([record["x"] for record in records]).astype(np.float32)
    activity_id = np.asarray([record["activity_id"] for record in records], dtype=np.int64)

    arrays = {
        "X": x,
        "pose": np.stack([record["pose"] for record in records]).astype(np.float32),
        "pose_mask": np.asarray([record["pose_mask"] for record in records], dtype=np.int64),
        "activity": one_hot(activity_id, len(label_maps["activity"])),
        "activity_id": activity_id,
        "activity_mask": np.asarray([record["activity_mask"] for record in records], dtype=np.int64),
        "location_id": np.asarray([record["location_id"] for record in records], dtype=np.int64),
        "location_mask": np.asarray([record["location_mask"] for record in records], dtype=np.int64),
        "environment_id": np.asarray([record["environment_id"] for record in records], dtype=np.int64),
        "environment_mask": np.asarray([record["environment_mask"] for record in records], dtype=np.int64),
        "subject_id": np.asarray([record["subject_id"] for record in records], dtype=np.int64),
        "subject_mask": np.asarray([record["subject_mask"] for record in records], dtype=np.int64),
    }
    return arrays


def write_dataset(
    out: Path,
    dataset: str,
    records: list[dict],
    label_maps: dict[str, dict[str, int]],
    *,
    window_frames: int,
    stride: int,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    metadata = np.asarray(
        [
            {
                "dataset": dataset,
                "sample_id": record["sample_id"],
                "split": record["split"],
                "window_start": record["window_start"],
            }
            for record in records
        ],
        dtype=object,
    )

    all_arrays = records_to_arrays(records, label_maps)
    np.savez_compressed(out / "x.npz", X=all_arrays["X"])
    np.savez_compressed(
        out / "y.npz",
        pose=all_arrays["pose"],
        pose_mask=all_arrays["pose_mask"],
        activity=all_arrays["activity"],
        activity_id=all_arrays["activity_id"],
        activity_mask=all_arrays["activity_mask"],
        location_id=all_arrays["location_id"],
        location_mask=all_arrays["location_mask"],
        environment_id=all_arrays["environment_id"],
        environment_mask=all_arrays["environment_mask"],
        subject_id=all_arrays["subject_id"],
        subject_mask=all_arrays["subject_mask"],
    )
    np.savez_compressed(out / "metadata.npz", metadata=metadata)

    split_counts = defaultdict(int)
    for record in records:
        split_counts[record["split"]] += 1

    normalization = {
        "mean": float(all_arrays["X"].mean()),
        "std": float(all_arrays["X"].std() + 1e-6),
    }
    stats = {
        "dataset": dataset,
        "num_samples": len(records),
        "splits": dict(sorted(split_counts.items())),
        "x_shape": list(all_arrays["X"].shape),
        "pose_shape": list(all_arrays["pose"].shape),
        "window_frames": window_frames,
        "stride": stride,
        "pose_joints": POSE_JOINTS,
    }
    manifest = {
        **stats,
        "files": {
            "x": "x.npz",
            "y": "y.npz",
            "metadata": "metadata.npz",
            "label_maps": "label_maps.json",
            "stats": "stats.json",
            "normalization": "normalization.json",
        },
    }

    (out / "label_maps.json").write_text(json.dumps(label_maps, indent=2, sort_keys=True))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2))
    return stats


def filter_rows(
    rows: list[dict],
    *,
    datasets: set[str] | None,
    max_samples_per_dataset: int | None,
) -> list[dict]:
    if datasets is not None:
        rows = [row for row in rows if row.get("dataset") in datasets]

    if max_samples_per_dataset is None:
        return rows

    seen = defaultdict(set)
    filtered = []
    for row in rows:
        dataset = row.get("dataset")
        sample_id = str(row.get("sample_id"))
        if len(seen[dataset]) >= max_samples_per_dataset and sample_id not in seen[dataset]:
            continue
        seen[dataset].add(sample_id)
        filtered.append(row)
    return filtered


def silver_to_gold(
    silver_path: str | Path,
    gold_dir: str | Path,
    *,
    datasets: set[str] | None = None,
    window_frames: int = 60,
    stride: int = 10,
    max_samples_per_dataset: int | None = None,
    force: bool = False,
) -> dict:
    # --- Idempotent check ---
    if not force and not is_s3_uri(gold_dir):
        summary_path = Path(gold_dir) / "summary.json"
        if summary_path.exists():
            try:
                cached = json.loads(summary_path.read_text())
                cached["skipped"] = True
                log.info("SKIP silver_to_gold: output exists at %s (%d samples)", gold_dir, cached.get("num_samples", 0))
                return cached
            except Exception:
                pass
    elif not force and is_s3_uri(gold_dir):
        try:
            bucket, prefix = parse_s3_uri(gold_dir)
            summary_key = f"{prefix.rstrip('/')}/summary.json"
            client = make_s3_client()
            resp = client.get_object(Bucket=bucket, Key=summary_key)
            cached = json.loads(resp["Body"].read().decode())
            cached["skipped"] = True
            log.info("SKIP silver_to_gold: S3 output exists at %s (%d samples)", gold_dir, cached.get("num_samples", 0))
            return cached
        except Exception:
            pass

    log.info("START silver_to_gold: silver=%s -> gold=%s (datasets=%s, window=%d, stride=%d)",
             silver_path, gold_dir, datasets, window_frames, stride)
    t0 = time.time()

    log.info("  Loading silver data from %s ...", silver_path)
    rows = load_silver(silver_path)
    log.info("  Loaded %d raw rows", len(rows))
    rows = filter_rows(
        rows,
        datasets=datasets,
        max_samples_per_dataset=max_samples_per_dataset,
    )
    grouped = group_silver_rows(rows)
    log.info("  Grouped into %d unique (dataset, sample) pairs", len(grouped))
    records_by_dataset, label_maps = build_gold_records(
        grouped,
        window_frames=window_frames,
        stride=stride,
    )

    if not records_by_dataset:
        raise RuntimeError(
            "No gold windows were created. Check dataset filter, window_frames, "
            "and whether silver rows contain enough timestamps."
        )

    if is_s3_uri(gold_dir):
        with tempfile.TemporaryDirectory(prefix="rfpose-gold-") as tmpdir:
            local_gold_dir = Path(tmpdir) / "gold"
            summary = silver_to_gold(
                silver_path,
                local_gold_dir,
                datasets=datasets,
                window_frames=window_frames,
                stride=stride,
                max_samples_per_dataset=max_samples_per_dataset,
            )
            upload_report = upload_s3_directory(local_gold_dir, gold_dir)
            summary["artifact_uri"] = str(gold_dir)
            summary["upload"] = upload_report

            summary_path = local_gold_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            upload_s3_file(summary_path, f"{str(gold_dir).rstrip('/')}/summary.json")
            return summary

    out = Path(gold_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats_by_dataset = {}
    for dataset, records in sorted(records_by_dataset.items()):
        dataset_out = out if len(records_by_dataset) == 1 else out / dataset
        log.info("  Writing dataset '%s': %d records -> %s", dataset, len(records), dataset_out)
        stats_by_dataset[dataset] = write_dataset(
            dataset_out,
            dataset,
            records,
            label_maps,
            window_frames=window_frames,
            stride=stride,
        )

    summary = {
        "datasets": stats_by_dataset,
        "num_datasets": len(stats_by_dataset),
        "num_samples": sum(stats["num_samples"] for stats in stats_by_dataset.values()),
        "window_frames": window_frames,
        "stride": stride,
        "label_maps": "label_maps.json",
        "artifact_uri": str(gold_dir),
    }
    (out / "label_maps.json").write_text(json.dumps(label_maps, indent=2, sort_keys=True))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    summary["skipped"] = False
    elapsed = time.time() - t0
    log.info("DONE silver_to_gold: %d datasets, %d total samples in %.1fs",
             summary["num_datasets"], summary["num_samples"], elapsed)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver-path", required=True)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument(
        "--datasets",
        default=os.getenv("RFPOSE_GOLD_DATASETS"),
        help="Comma-separated dataset filter, e.g. wimans,mmfi,wipose.",
    )
    parser.add_argument("--window-frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    args = parser.parse_args()

    print(
        json.dumps(
            silver_to_gold(
                args.silver_path,
                args.gold_dir,
                datasets=parse_dataset_filter(args.datasets),
                window_frames=args.window_frames,
                stride=args.stride,
                max_samples_per_dataset=args.max_samples_per_dataset,
            ),
            indent=2,
        )
    )
