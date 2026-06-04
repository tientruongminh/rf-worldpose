"""Silver → Gold ETL: read catalog + .npy CSI files, cut windows, pack NPZ.

Silver input: directory with catalog.parquet + csi/<dataset>/<sample>.npy
Gold output: directory with per-dataset x.npz, y.npz, metadata.npz
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import logging
import os
import time
import tempfile

import numpy as np

try:
    import polars as pl
except Exception:
    pl = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

POSE_JOINTS = [
    "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
# Helpers (kept from original)
# ---------------------------------------------------------------------------

def is_s3_uri(value) -> bool:
    return str(value).startswith("s3://")

def parse_s3_uri(uri) -> tuple[str, str]:
    value = str(uri)
    if not value.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {value}")
    without_scheme = value[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix.strip("/")

def make_s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 required") from exc
    endpoint_url = os.getenv("S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL") or "http://207.180.243.242:9000"
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    kwargs = {"endpoint_url": endpoint_url}
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)

def upload_s3_file(local_path: Path, s3_uri) -> dict:
    bucket, key = parse_s3_uri(s3_uri)
    client = make_s3_client()
    client.upload_file(str(local_path), bucket, key)
    return {"uri": f"s3://{bucket}/{key}", "bytes": local_path.stat().st_size}

def upload_s3_directory(local_dir: Path, s3_uri) -> dict:
    bucket, prefix = parse_s3_uri(s3_uri)
    prefix = prefix.strip("/")
    client = make_s3_client()
    count, total_bytes = 0, 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        key = f"{prefix}/{path.relative_to(local_dir).as_posix()}" if prefix else path.relative_to(local_dir).as_posix()
        client.upload_file(str(path), bucket, key)
        count += 1
        total_bytes += path.stat().st_size
    return {"uri": f"s3://{bucket}/{prefix}", "object_count": count, "total_bytes": total_bytes}


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


def one_hot(ids: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.zeros((len(ids), num_classes), dtype=np.float32)
    valid = ids >= 0
    if num_classes > 0 and valid.any():
        y[np.where(valid)[0], ids[valid]] = 1.0
    return y


# ---------------------------------------------------------------------------
# Load silver catalog
# ---------------------------------------------------------------------------

def load_catalog(silver_dir: str | Path) -> list[dict]:
    silver_dir = Path(silver_dir)
    parquet_path = silver_dir / "catalog.parquet"
    jsonl_path = silver_dir / "catalog.jsonl"

    if parquet_path.exists() and pl is not None:
        log.info("  [load_catalog] Reading %s ...", parquet_path)
        rows = pl.read_parquet(parquet_path).to_dicts()
    elif jsonl_path.exists():
        log.info("  [load_catalog] Reading %s ...", jsonl_path)
        lines = jsonl_path.read_text().splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    else:
        raise FileNotFoundError(f"No catalog found in {silver_dir}")

    log.info("  [load_catalog] Loaded %d samples", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Window cutting & gold record building
# ---------------------------------------------------------------------------

def make_windows(csi: np.ndarray, window_frames: int, stride: int) -> list[tuple[int, np.ndarray]]:
    """Cut sliding windows along time axis (axis=1)."""
    T = csi.shape[1]
    if T < window_frames:
        return []
    windows = []
    for start in range(0, T - window_frames + 1, stride):
        windows.append((start, csi[:, start:start + window_frames, ...]))
    return windows


def build_gold_from_catalog(
    catalog: list[dict],
    silver_dir: Path,
    *,
    window_frames: int,
    stride: int,
    datasets_filter: set[str] | None = None,
    max_samples_per_dataset: int | None = None,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:

    label_maps = {"activity": {}, "location": {}, "environment": {}, "subject": {}}
    records_by_dataset = defaultdict(list)

    # Filter catalog
    if datasets_filter:
        catalog = [r for r in catalog if r.get("dataset") in datasets_filter]

    # Apply max_samples_per_dataset
    if max_samples_per_dataset is not None:
        seen = defaultdict(int)
        filtered = []
        for row in catalog:
            ds = row.get("dataset")
            if seen[ds] < max_samples_per_dataset:
                filtered.append(row)
                seen[ds] += 1
        catalog = filtered

    total = len(catalog)
    log.info("  [build_gold] Processing %d samples (window=%d, stride=%d) ...", total, window_frames, stride)
    t0 = time.time()
    skipped_short = 0
    skipped_load = 0

    for idx, row in enumerate(catalog):
        dataset = row["dataset"]
        sample_id = row["sample_id"]
        csi_path = silver_dir / row["csi_path"]

        # Load CSI tensor
        try:
            csi = np.load(csi_path).astype(np.float32)
        except Exception as e:
            log.warning("  [build_gold] Failed to load %s: %s", csi_path, e)
            skipped_load += 1
            continue

        # Cut windows
        windows = make_windows(csi, window_frames, stride)
        if not windows:
            skipped_short += 1
            continue

        # Parse pose
        pose_data = row.get("pose")
        if pose_data is not None:
            if isinstance(pose_data, str):
                pose_data = json.loads(pose_data)
            pose = np.asarray(pose_data, dtype=np.float32).reshape(13, 3)
            pose_mask = 1
        else:
            pose = np.zeros((13, 3), dtype=np.float32)
            pose_mask = 0

        # Labels
        sample_key = f"{dataset}:{sample_id}"
        split = normalize_split(row.get("split"), sample_key)
        activity_id = label_index(row.get("activity"), label_maps["activity"])
        location_id = label_index(row.get("location_key") or row.get("location"), label_maps["location"])
        environment_id = label_index(row.get("environment_key") or row.get("environment"), label_maps["environment"])
        subject_id = label_index(row.get("subject_key") or row.get("subject"), label_maps["subject"])

        for window_start, window_x in windows:
            records_by_dataset[dataset].append({
                "split": split,
                "sample_id": sample_id,
                "window_start": window_start,
                "x": window_x,
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
            })

        if (idx + 1) % 500 == 0 or idx + 1 == total:
            total_windows = sum(len(v) for v in records_by_dataset.values())
            log.info("  [build_gold] %d/%d samples (%.0f%%) | %d windows | %d short-skipped | %.1fs",
                     idx + 1, total, (idx + 1) / total * 100, total_windows, skipped_short, time.time() - t0)

    total_windows = sum(len(v) for v in records_by_dataset.values())
    log.info("  [build_gold] DONE: %d windows from %d datasets | %d short-skipped | %d load-errors | %.1fs",
             total_windows, len(records_by_dataset), skipped_short, skipped_load, time.time() - t0)

    return dict(records_by_dataset), label_maps


# ---------------------------------------------------------------------------
# Write gold NPZ files
# ---------------------------------------------------------------------------

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
    log.info("    [write/%s] Stacking %d records ...", dataset, len(records))
    t0 = time.time()

    x = np.stack([r["x"] for r in records]).astype(np.float32)
    activity_id = np.asarray([r["activity_id"] for r in records], dtype=np.int64)

    log.info("    [write/%s] X shape=%s (%.1f MB)", dataset, x.shape, x.nbytes / 1024 / 1024)

    np.savez_compressed(out / "x.npz", X=x)
    np.savez_compressed(out / "y.npz",
        pose=np.stack([r["pose"] for r in records]).astype(np.float32),
        pose_mask=np.asarray([r["pose_mask"] for r in records], dtype=np.int64),
        activity=one_hot(activity_id, len(label_maps["activity"])),
        activity_id=activity_id,
        activity_mask=np.asarray([r["activity_mask"] for r in records], dtype=np.int64),
        location_id=np.asarray([r["location_id"] for r in records], dtype=np.int64),
        location_mask=np.asarray([r["location_mask"] for r in records], dtype=np.int64),
        environment_id=np.asarray([r["environment_id"] for r in records], dtype=np.int64),
        environment_mask=np.asarray([r["environment_mask"] for r in records], dtype=np.int64),
        subject_id=np.asarray([r["subject_id"] for r in records], dtype=np.int64),
        subject_mask=np.asarray([r["subject_mask"] for r in records], dtype=np.int64),
    )
    np.savez_compressed(out / "metadata.npz", metadata=np.asarray([
        {"dataset": dataset, "sample_id": r["sample_id"], "split": r["split"], "window_start": r["window_start"]}
        for r in records
    ], dtype=object))

    split_counts = defaultdict(int)
    for r in records:
        split_counts[r["split"]] += 1

    normalization = {"mean": float(x.mean()), "std": float(x.std() + 1e-6)}
    stats = {
        "dataset": dataset, "num_samples": len(records),
        "splits": dict(sorted(split_counts.items())),
        "x_shape": list(x.shape),
        "pose_shape": [len(records), 13, 3],
        "window_frames": window_frames, "stride": stride,
        "pose_joints": POSE_JOINTS,
    }
    manifest = {**stats, "files": {"x": "x.npz", "y": "y.npz", "metadata": "metadata.npz",
                                    "label_maps": "label_maps.json", "stats": "stats.json", "normalization": "normalization.json"}}
    (out / "label_maps.json").write_text(json.dumps(label_maps, indent=2, sort_keys=True))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2))
    del x
    log.info("    [write/%s] DONE in %.1fs", dataset, time.time() - t0)
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def silver_to_gold(
    silver_dir: str | Path,
    gold_dir: str | Path,
    *,
    datasets: set[str] | None = None,
    window_frames: int = 60,
    stride: int = 10,
    max_samples_per_dataset: int | None = None,
    force: bool = False,
) -> dict:
    # Idempotent check
    if not force and not is_s3_uri(gold_dir):
        summary_path = Path(gold_dir) / "summary.json"
        if summary_path.exists():
            try:
                cached = json.loads(summary_path.read_text())
                cached["skipped"] = True
                log.info("SKIP silver_to_gold: output exists at %s", gold_dir)
                return cached
            except Exception:
                pass

    log.info("START silver_to_gold: silver=%s -> gold=%s (window=%d, stride=%d)", silver_dir, gold_dir, window_frames, stride)
    t0 = time.time()

    silver_dir = Path(silver_dir)
    catalog = load_catalog(silver_dir)

    records_by_dataset, label_maps = build_gold_from_catalog(
        catalog, silver_dir,
        window_frames=window_frames, stride=stride,
        datasets_filter=datasets,
        max_samples_per_dataset=max_samples_per_dataset,
    )

    if not records_by_dataset:
        raise RuntimeError(
            "No gold windows created. Check window_frames vs dataset timesteps. "
            f"Current window_frames={window_frames}."
        )

    # Handle S3 output
    if is_s3_uri(gold_dir):
        with tempfile.TemporaryDirectory(prefix="rfpose-gold-") as tmpdir:
            local_gold = Path(tmpdir) / "gold"
            summary = silver_to_gold(
                silver_dir, local_gold,
                datasets=datasets, window_frames=window_frames,
                stride=stride, max_samples_per_dataset=max_samples_per_dataset,
            )
            upload_report = upload_s3_directory(local_gold, gold_dir)
            summary["upload"] = upload_report
            summary_path = local_gold / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            upload_s3_file(summary_path, f"{str(gold_dir).rstrip('/')}/summary.json")
            return summary

    out = Path(gold_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats_by_dataset = {}
    total_ds = len(records_by_dataset)
    for ds_idx, (dataset, records) in enumerate(sorted(records_by_dataset.items()), 1):
        dataset_out = out if total_ds == 1 else out / dataset
        log.info("  [%d/%d] Writing '%s': %d windows -> %s", ds_idx, total_ds, dataset, len(records), dataset_out)
        stats_by_dataset[dataset] = write_dataset(
            dataset_out, dataset, records, label_maps,
            window_frames=window_frames, stride=stride,
        )

    summary = {
        "datasets": stats_by_dataset,
        "num_datasets": len(stats_by_dataset),
        "num_samples": sum(s["num_samples"] for s in stats_by_dataset.values()),
        "window_frames": window_frames, "stride": stride,
        "label_maps": "label_maps.json",
        "artifact_uri": str(gold_dir),
    }
    (out / "label_maps.json").write_text(json.dumps(label_maps, indent=2, sort_keys=True))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    summary["skipped"] = False
    elapsed = time.time() - t0
    log.info("DONE silver_to_gold: %d datasets, %d total windows in %.1fs",
             summary["num_datasets"], summary["num_samples"], elapsed)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("RFPOSE_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver-dir", required=True, help="Silver directory with catalog + csi/")
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--datasets", default=os.getenv("RFPOSE_GOLD_DATASETS"))
    parser.add_argument("--window-frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(silver_to_gold(
        args.silver_dir, args.gold_dir,
        datasets=parse_dataset_filter(args.datasets),
        window_frames=args.window_frames, stride=args.stride,
        max_samples_per_dataset=args.max_samples_per_dataset,
    ), indent=2))
