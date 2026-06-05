"""Silver-Unified → Gold ETL: read unified catalog, load CSI from Silver,
flatten+pad+normalize on-the-fly, cut windows, write per-dataset NPZ.

The unified catalog (catalog-only mode) provides n_padded and normalization
stats. CSI files are read from the original Silver directory and transformed
during windowing. Processes ONE dataset at a time to avoid OOM.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import gc
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

MAX_WINDOWS_PER_SAMPLE = 10


# ---------------------------------------------------------------------------
# Helpers
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
    if split is not None and split != "":
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
    if label is None or label == "":
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
# Window cutting
# ---------------------------------------------------------------------------

def make_windows(csi: np.ndarray, window_frames: int, stride: int) -> list[tuple[int, np.ndarray]]:
    T = csi.shape[1]
    if T < window_frames:
        return []
    effective_stride = stride
    max_possible = (T - window_frames) // stride + 1
    if max_possible > MAX_WINDOWS_PER_SAMPLE:
        effective_stride = max(1, (T - window_frames) // (MAX_WINDOWS_PER_SAMPLE - 1))
    windows = []
    for start in range(0, T - window_frames + 1, effective_stride):
        windows.append((start, csi[:, start:start + window_frames, ...]))
        if len(windows) >= MAX_WINDOWS_PER_SAMPLE:
            break
    return windows


# ---------------------------------------------------------------------------
# Process one dataset at a time
# ---------------------------------------------------------------------------

def flatten_pad_normalize(csi: np.ndarray, n_padded: int, c_unified: int = 2) -> np.ndarray:
    """Flatten spatial dims, pad C and N, z-score normalize. [C,T,...] -> [c_unified,T,n_padded]."""
    C, T = csi.shape[0], csi.shape[1]
    N = int(np.prod(csi.shape[2:]))
    flat = csi.reshape(C, T, N).astype(np.float32)

    out = np.zeros((c_unified, T, n_padded), dtype=np.float32)
    c_copy = min(C, c_unified)
    n_copy = min(N, n_padded)
    out[:c_copy, :, :n_copy] = flat[:c_copy, :, :n_copy]

    for ch in range(min(c_copy, c_unified)):
        mean = float(out[ch].mean())
        std = float(out[ch].std() + 1e-8)
        out[ch] = (out[ch] - mean) / std
    return out


def process_one_dataset(
    dataset_name: str,
    samples: list[dict],
    silver_dir: Path,
    out_dir: Path,
    label_maps: dict[str, dict[str, int]],
    *,
    window_frames: int,
    stride: int,
    n_padded: int,
    c_unified: int = 2,
) -> dict | None:
    """Process a single dataset: load CSI, flatten+pad+normalize, cut windows, write NPZ."""
    log.info("  [gold/%s] Processing %d samples (n_padded=%d) ...", dataset_name, len(samples), n_padded)
    t0 = time.time()

    records = []
    skipped_short = 0
    skipped_load = 0

    for idx, row in enumerate(samples):
        sample_id = row["sample_id"]
        csi_rel = row.get("original_csi_path") or row.get("csi_path")
        csi_path = silver_dir / csi_rel

        try:
            csi = np.load(csi_path, mmap_mode="r")
            csi_data = csi[:]
            del csi
        except Exception as e:
            skipped_load += 1
            continue

        # Flatten, pad, normalize on-the-fly
        csi_unified = flatten_pad_normalize(csi_data, n_padded, c_unified)
        del csi_data

        windows = make_windows(csi_unified, window_frames, stride)
        del csi_unified
        if not windows:
            skipped_short += 1
            continue

        # Parse pose
        pose_data = row.get("pose")
        if pose_data is not None and pose_data != "":
            if isinstance(pose_data, str):
                pose_data = json.loads(pose_data)
            pose = np.asarray(pose_data, dtype=np.float32).reshape(13, 3)
            pose_mask = 1
        else:
            pose = np.zeros((13, 3), dtype=np.float32)
            pose_mask = 0

        sample_key = f"{dataset_name}:{sample_id}"
        split = normalize_split(row.get("split"), sample_key)
        activity_id = label_index(row.get("activity"), label_maps["activity"])
        location_id = label_index(row.get("location_key") or row.get("location"), label_maps["location"])
        environment_id = label_index(row.get("environment_key") or row.get("environment"), label_maps["environment"])
        subject_id = label_index(row.get("subject_key") or row.get("subject"), label_maps["subject"])

        for window_start, window_x in windows:
            records.append({
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
        del windows

        if (idx + 1) % 1000 == 0:
            log.info("    [gold/%s] %d/%d samples | %d windows | %.1fs",
                     dataset_name, idx + 1, len(samples), len(records), time.time() - t0)

    if not records:
        log.info("  [gold/%s] No windows generated (short=%d, load_err=%d)", dataset_name, skipped_short, skipped_load)
        return None

    log.info("  [gold/%s] %d windows from %d samples (short=%d, load_err=%d) in %.1fs. Writing NPZ ...",
             dataset_name, len(records), len(samples), skipped_short, skipped_load, time.time() - t0)

    # Write NPZ
    out_dir.mkdir(parents=True, exist_ok=True)
    t_write = time.time()

    x = np.stack([r["x"] for r in records]).astype(np.float32)
    activity_id_arr = np.asarray([r["activity_id"] for r in records], dtype=np.int64)
    log.info("    [gold/%s] X shape=%s (%.1f MB)", dataset_name, x.shape, x.nbytes / 1024 / 1024)

    np.savez_compressed(out_dir / "x.npz", X=x)
    del x

    np.savez_compressed(out_dir / "y.npz",
        pose=np.stack([r["pose"] for r in records]).astype(np.float32),
        pose_mask=np.asarray([r["pose_mask"] for r in records], dtype=np.int64),
        activity=one_hot(activity_id_arr, len(label_maps["activity"])),
        activity_id=activity_id_arr,
        activity_mask=np.asarray([r["activity_mask"] for r in records], dtype=np.int64),
        location_id=np.asarray([r["location_id"] for r in records], dtype=np.int64),
        location_mask=np.asarray([r["location_mask"] for r in records], dtype=np.int64),
        environment_id=np.asarray([r["environment_id"] for r in records], dtype=np.int64),
        environment_mask=np.asarray([r["environment_mask"] for r in records], dtype=np.int64),
        subject_id=np.asarray([r["subject_id"] for r in records], dtype=np.int64),
        subject_mask=np.asarray([r["subject_mask"] for r in records], dtype=np.int64),
    )

    np.savez_compressed(out_dir / "metadata.npz", metadata=np.asarray([
        {"dataset": dataset_name, "sample_id": r["sample_id"], "split": r["split"], "window_start": r["window_start"]}
        for r in records
    ], dtype=object))

    split_counts = defaultdict(int)
    for r in records:
        split_counts[r["split"]] += 1

    stats = {
        "dataset": dataset_name, "num_samples": len(records),
        "splits": dict(sorted(split_counts.items())),
        "x_shape": [len(records)] + list(records[0]["x"].shape),
        "pose_shape": [len(records), 13, 3],
        "window_frames": window_frames, "stride": stride,
        "pose_joints": POSE_JOINTS,
    }

    # Compute normalization from a subsample to avoid re-loading x.npz
    all_means = []
    all_stds = []
    for r in records[:min(1000, len(records))]:
        all_means.append(float(r["x"].mean()))
        all_stds.append(float(r["x"].std()))
    normalization = {"mean": float(np.mean(all_means)), "std": float(np.mean(all_stds) + 1e-6)}

    manifest = {**stats, "files": {"x": "x.npz", "y": "y.npz", "metadata": "metadata.npz",
                                    "label_maps": "label_maps.json", "stats": "stats.json", "normalization": "normalization.json"}}
    (out_dir / "label_maps.json").write_text(json.dumps(label_maps, indent=2, sort_keys=True))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "normalization.json").write_text(json.dumps(normalization, indent=2))

    del records
    gc.collect()

    log.info("    [gold/%s] Written in %.1fs", dataset_name, time.time() - t_write)
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def silver_to_gold(
    unified_dir: str | Path,
    gold_dir: str | Path,
    *,
    silver_dir: str | Path | None = None,
    datasets: set[str] | None = None,
    window_frames: int = 60,
    stride: int = 10,
    max_samples_per_dataset: int | None = None,
    force: bool = False,
) -> dict:
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

    log.info("START silver_to_gold: unified=%s -> gold=%s (window=%d, stride=%d, max_win/sample=%d)",
             unified_dir, gold_dir, window_frames, stride, MAX_WINDOWS_PER_SAMPLE)
    t0 = time.time()

    unified_dir = Path(unified_dir)
    catalog = load_catalog(unified_dir)

    # Determine silver_dir for loading original .npy files
    if silver_dir is not None:
        actual_silver_dir = Path(silver_dir)
    else:
        report_path = unified_dir / "quality_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            actual_silver_dir = Path(report.get("silver_dir", str(unified_dir)))
        else:
            actual_silver_dir = unified_dir
    log.info("  Silver CSI dir: %s", actual_silver_dir)

    # Get n_padded from catalog
    n_padded_values = set()
    for row in catalog:
        np_val = row.get("n_padded")
        if np_val is not None and np_val > 0:
            n_padded_values.add(int(np_val))
    n_padded = max(n_padded_values) if n_padded_values else None
    c_unified = 2
    if n_padded:
        log.info("  Using n_padded=%d, c_unified=%d from unified catalog", n_padded, c_unified)

    # Filter
    if datasets:
        catalog = [r for r in catalog if r.get("dataset") in datasets]
    if max_samples_per_dataset is not None:
        seen = defaultdict(int)
        filtered = []
        for row in catalog:
            ds = row.get("dataset")
            if seen[ds] < max_samples_per_dataset:
                filtered.append(row)
                seen[ds] += 1
        catalog = filtered

    # Group by dataset
    by_dataset = defaultdict(list)
    for row in catalog:
        by_dataset[row.get("dataset", "unknown")].append(row)
    del catalog

    log.info("  %d datasets to process: %s",
             len(by_dataset), {k: len(v) for k, v in sorted(by_dataset.items())})

    if is_s3_uri(gold_dir):
        with tempfile.TemporaryDirectory(prefix="rfpose-gold-") as tmpdir:
            local_gold = Path(tmpdir) / "gold"
            summary = silver_to_gold(
                unified_dir, local_gold,
                silver_dir=silver_dir, datasets=datasets,
                window_frames=window_frames, stride=stride,
                max_samples_per_dataset=max_samples_per_dataset,
            )
            upload_report = upload_s3_directory(local_gold, gold_dir)
            summary["upload"] = upload_report
            summary_path = local_gold / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            upload_s3_file(summary_path, f"{str(gold_dir).rstrip('/')}/summary.json")
            return summary

    out = Path(gold_dir)
    out.mkdir(parents=True, exist_ok=True)

    label_maps = {"activity": {}, "location": {}, "environment": {}, "subject": {}}
    stats_by_dataset = {}
    total_ds = len(by_dataset)

    for ds_idx, (dataset_name, samples) in enumerate(sorted(by_dataset.items()), 1):
        log.info("[%d/%d] Dataset '%s': %d samples", ds_idx, total_ds, dataset_name, len(samples))
        dataset_out = out / dataset_name
        stats = process_one_dataset(
            dataset_name, samples, actual_silver_dir, dataset_out, label_maps,
            window_frames=window_frames, stride=stride,
            n_padded=n_padded, c_unified=c_unified,
        )
        if stats:
            stats_by_dataset[dataset_name] = stats
        gc.collect()

    if not stats_by_dataset:
        raise RuntimeError("No gold windows created from any dataset.")

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
    parser.add_argument("--unified-dir", required=True, help="Dir with unified catalog")
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--silver-dir", default=None, help="Dir with original .npy files (defaults to quality_report.silver_dir)")
    parser.add_argument("--datasets", default=os.getenv("RFPOSE_GOLD_DATASETS"))
    parser.add_argument("--window-frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(silver_to_gold(
        args.unified_dir, args.gold_dir,
        silver_dir=args.silver_dir,
        datasets=parse_dataset_filter(args.datasets),
        window_frames=args.window_frames, stride=args.stride,
        max_samples_per_dataset=args.max_samples_per_dataset,
    ), indent=2))
