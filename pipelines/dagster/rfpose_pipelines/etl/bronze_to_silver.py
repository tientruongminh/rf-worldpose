"""Bronze → Silver ETL: one row per sample, CSI saved as .npy binary.

Silver output is a directory containing:
  - catalog.parquet: metadata for each sample (1 row = 1 sample)
  - csi/<dataset>/<sample_id>.npy: CSI tensor per sample
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator
import argparse
import json
import logging
import time
import os
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None

from datasets.mmfi_reader import (
    index_mmfi_wifi_csi_samples,
    load_mmfi_csi_mat,
    load_mmfi_pose_npy,
    mmfi_frame_to_index,
)
from datasets.person_wifi_reader import (
    index_person_wifi_samples,
    load_person_wifi_csi_mat,
    load_person_wifi_keypoint_npy,
)
from datasets.uthar_reader import UT_HAR_ACTIONS, load_uthar_arrays
from datasets.wiar_reader import index_wiar_samples, load_wiar_sample
from datasets.wimans_reader import iter_wimans_samples, load_wimans_amp
from datasets.wipose_reader import index_wipose_samples, load_wipose_mat


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# Common pose mapping
# ---------------------------------------------------------------------------

COMMON_POSE_JOINTS = [
    "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

MMFI_TO_COMMON = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
WIPOSE_TO_COMMON = [0, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
PERSON_WIFI_TO_COMMON = [0, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

ACTIVITY_ALIASES = {
    "sitdown": "sit_down", "sit_down": "sit_down",
    "standup": "stand_up", "stand_up": "stand_up",
    "pick_up": "pick_up", "picking_up_things": "pick_up",
    "jump": "jump", "jumping_up": "jump",
    "throw": "throw", "throwing_left_side": "throw",
    "throwing_right_side": "throw", "high_throw": "throw",
    "wave": "wave", "waving_hand_left": "wave",
    "waving_hand_right": "wave", "horizontal_arm_wave": "wave",
    "high_arm_wave": "wave", "two_hands_wave": "wave",
    "forward_kick": "kick", "side_kick": "kick",
    "kicking_left_side": "kick", "kicking_right_side": "kick",
    "squat": "squat", "crouch": "squat",
    "bend": "bend", "bowing": "bend",
    "walk": "walk", "run": "run", "fall": "fall", "lie_down": "lie_down",
}


def normalize_activity(activity) -> str | None:
    if activity is None:
        return None
    normalized = str(activity).strip().lower().replace(" ", "_").replace("-", "_")
    return ACTIVITY_ALIASES.get(normalized, normalized)


def namespace_key(dataset: str | None, value) -> str | None:
    if dataset is None or value is None:
        return None
    return f"{dataset}:{value}"


def select_common_pose_joints(pose, indices: list[int], *, dataset: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2:
        raise ValueError(f"{dataset} pose must be 2D [joints, dims], got {pose.shape}")
    if pose.shape[0] <= max(indices):
        raise ValueError(f"{dataset} pose has {pose.shape[0]} joints, need index {max(indices)}")
    return pose[indices]


def limit_samples(samples: Iterable, max_samples: int | None) -> Iterable:
    if max_samples is None:
        yield from samples
        return
    for idx, sample in enumerate(samples):
        if idx >= max_samples:
            break
        yield sample


def _safe_sample_id(raw_id: str) -> str:
    """Sanitize sample_id for use as a filename."""
    return str(raw_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Per-dataset converters: yield (catalog_row, csi_tensor) per sample
# ---------------------------------------------------------------------------

def convert_wimans(root: Path, max_samples: int | None = None) -> Iterator[tuple[dict, np.ndarray]]:
    samples = iter_wimans_samples(root, single_person_only=False, include_empty_room=True)
    for sample in limit_samples(samples, max_samples):
        amp = load_wimans_amp(sample["amp_path"])
        if amp.ndim != 4:
            log.warning("[wimans] Skipping bad shape %s: %s", sample["amp_path"], amp.shape)
            continue
        T, tx, rx, sc = amp.shape
        activity = sample["activities"][0] if sample["activities"] else None
        location = sample["locations"][0] if sample["locations"] else None
        csi = amp.astype(np.float32)[np.newaxis]  # [1, T, tx, rx, sc]
        row = dict(
            dataset="wimans", sample_id=sample["label"],
            source_file=str(sample["amp_path"]),
            split=None, activity=normalize_activity(activity),
            subject=None, environment=sample["environment"], location=location,
            num_users=sample["num_users"],
            has_pose=False, pose=None, pose_dim=None,
            antenna_layout="txrx_pair", has_phase=False,
            n_tx=tx, n_rx=rx, n_antennas=None,
            n_subcarriers=sc, n_timesteps=T,
        )
        yield row, csi


def convert_person_wifi(root: Path, max_samples: int | None = None) -> Iterator[tuple[dict, np.ndarray]]:
    for split in ("train", "test"):
        split_dir = root / f"{split}_data"
        if not split_dir.exists():
            continue
        samples = list(index_person_wifi_samples(root, split=split, single_person_only=False))
        for sample in limit_samples(iter(samples), max_samples):
            csi_raw = load_person_wifi_csi_mat(sample["csi_path"])
            keypoint = load_person_wifi_keypoint_npy(sample["keypoint_path"], single_person_only=False)
            try:
                common_pose = select_common_pose_joints(keypoint[0], PERSON_WIFI_TO_COMMON, dataset="person_in_wifi_3d")
            except (ValueError, IndexError) as e:
                log.warning("[person_wifi] Skipping %s: %s", sample["name"], e)
                continue
            amp = np.abs(csi_raw).astype(np.float32)
            phase = np.angle(csi_raw).astype(np.float32)
            ant1, ant2, sc, T = amp.shape
            csi = np.stack([
                amp.transpose(3, 0, 1, 2),   # [T, ant1, ant2, sc]
                phase.transpose(3, 0, 1, 2),
            ])  # [2, T, ant1, ant2, sc]
            row = dict(
                dataset="person_in_wifi_3d", sample_id=sample["name"],
                source_file=str(sample["csi_path"]),
                split=split, activity=None,
                subject=None, environment=None, location=None,
                num_users=sample["num_people"],
                has_pose=True, pose=common_pose.tolist(), pose_dim=int(common_pose.shape[1]),
                antenna_layout="txrx_pair", has_phase=True,
                n_tx=ant1, n_rx=ant2, n_antennas=None,
                n_subcarriers=sc, n_timesteps=T,
            )
            yield row, csi


def convert_wipose(root: Path, max_samples: int | None = None) -> Iterator[tuple[dict, np.ndarray]]:
    samples = list(index_wipose_samples(root))
    for sample in limit_samples(iter(samples), max_samples):
        csi_raw, pose = load_wipose_mat(sample["path"])
        try:
            common_pose = select_common_pose_joints(pose, WIPOSE_TO_COMMON, dataset="wipose")
        except (ValueError, IndexError) as e:
            log.warning("[wipose] Skipping %s: %s", sample["path"].stem, e)
            continue
        tx, rx, sc, T, _ch = csi_raw.shape
        amp = np.abs(csi_raw[:, :, :, :, 0]).astype(np.float32)
        csi = amp.transpose(3, 0, 1, 2)[np.newaxis]  # [1, T, tx, rx, sc]
        row = dict(
            dataset="wipose", sample_id=sample["path"].stem,
            source_file=str(sample["path"]),
            split=sample.get("split"), activity=normalize_activity(sample.get("action")),
            subject=sample.get("subject"), environment=None, location=None,
            num_users=None,
            has_pose=True, pose=common_pose.tolist(), pose_dim=int(common_pose.shape[1]),
            antenna_layout="txrx_pair", has_phase=False,
            n_tx=tx, n_rx=rx, n_antennas=None,
            n_subcarriers=sc, n_timesteps=T,
        )
        yield row, csi


def convert_mmfi(root: Path, max_samples: int | None = None) -> Iterator[tuple[dict, np.ndarray]]:
    samples = list(index_mmfi_wifi_csi_samples(root, require_pose=False))
    for sample in limit_samples(iter(samples), max_samples):
        amp_raw, phase_raw = load_mmfi_csi_mat(sample["path"])
        antenna_count, sc, T = amp_raw.shape
        common_pose = None
        if sample["pose_path"] is not None:
            try:
                pose_index = mmfi_frame_to_index(sample["frame"])
                poses = load_mmfi_pose_npy(sample["pose_path"])
                common_pose = select_common_pose_joints(poses[pose_index], MMFI_TO_COMMON, dataset="mmfi")
            except (ValueError, IndexError) as e:
                log.warning("[mmfi] Pose error %s: %s", sample["path"], e)
        amp = amp_raw.astype(np.float32).transpose(2, 0, 1)      # [T, antenna, sc]
        phase = phase_raw.astype(np.float32).transpose(2, 0, 1)
        csi = np.stack([amp, phase])  # [2, T, antenna, sc]
        row = dict(
            dataset="mmfi", sample_id=_safe_sample_id(str(sample["path"].parent.parent)),
            source_file=str(sample["path"]),
            split=None, activity=normalize_activity(sample.get("action")),
            subject=sample.get("subject"), environment=sample.get("environment"), location=None,
            num_users=None,
            has_pose=common_pose is not None,
            pose=common_pose.tolist() if common_pose is not None else None,
            pose_dim=int(common_pose.shape[1]) if common_pose is not None else None,
            antenna_layout="antenna", has_phase=True,
            n_tx=None, n_rx=None, n_antennas=antenna_count,
            n_subcarriers=sc, n_timesteps=T,
        )
        yield row, csi


def convert_uthar(root: Path, max_samples: int | None = None) -> Iterator[tuple[dict, np.ndarray]]:
    for split in ("train", "val", "test"):
        x_all, y_all = load_uthar_arrays(root, split)
        sample_count = len(y_all) if max_samples is None else min(len(y_all), max_samples)
        for idx in range(sample_count):
            sample = x_all[idx].reshape(250, 3, 30).astype(np.float32)
            label = int(y_all[idx])
            csi = sample[np.newaxis]  # [1, T=250, antenna=3, sc=30]
            row = dict(
                dataset="uthar", sample_id=f"{split}_{idx}",
                source_file=str(root / "data" / f"X_{split}.csv"),
                split=split, activity=normalize_activity(UT_HAR_ACTIONS.get(label, f"unknown_{label}")),
                activity_id=label,
                subject=None, environment=None, location=None,
                num_users=None,
                has_pose=False, pose=None, pose_dim=None,
                antenna_layout="antenna", has_phase=False,
                n_tx=None, n_rx=None, n_antennas=3,
                n_subcarriers=30, n_timesteps=250,
            )
            yield row, csi


def convert_wiar(root: Path, max_samples: int | None = None) -> Iterator[tuple[dict, np.ndarray]]:
    samples = list(index_wiar_samples(root))
    for sample in limit_samples(iter(samples), max_samples):
        loaded = load_wiar_sample(sample["path"])
        csi_raw = loaded["csi"]
        T, sc, rx, tx = csi_raw.shape
        amp = np.abs(csi_raw).astype(np.float32)
        phase = np.angle(csi_raw).astype(np.float32)
        amp_t = amp.transpose(0, 2, 3, 1)    # [T, rx, tx, sc]
        phase_t = phase.transpose(0, 2, 3, 1)
        csi = np.stack([amp_t, phase_t])  # [2, T, rx, tx, sc]
        row = dict(
            dataset="wiar",
            sample_id=f"{sample['volunteer']}_{sample['activity_id']}_{sample['sample_id']}",
            source_file=str(sample["path"]),
            split=None, activity=normalize_activity(sample.get("activity_name")),
            activity_id=sample.get("activity_id"),
            subject=sample.get("volunteer"), environment=None, location=None,
            num_users=None,
            has_pose=False, pose=None, pose_dim=None,
            antenna_layout="txrx_pair", has_phase=True,
            n_tx=tx, n_rx=rx, n_antennas=None,
            n_subcarriers=sc, n_timesteps=T,
        )
        yield row, csi


# ---------------------------------------------------------------------------
# Dataset roots & converter registry
# ---------------------------------------------------------------------------

def existing_dataset_roots(bronze_root: str | Path) -> dict[str, Path]:
    root = Path(bronze_root)
    return {
        "wimans": root / "WiMANS",
        "person_in_wifi_3d": root / "wifipose_dataset",
        "wipose": root / "Wi-Pose" / "Wi-Pose",
        "mmfi": root / "MMFi_Dataset" / "MMFi_Dataset",
        "uthar": root / "UT_HAR",
        "wiar": root / "WiAR-master" / "WiAR-master" / "data" / "data",
    }

CONVERTERS = {
    "wimans": convert_wimans,
    "person_in_wifi_3d": convert_person_wifi,
    "wipose": convert_wipose,
    "mmfi": convert_mmfi,
    "uthar": convert_uthar,
    "wiar": convert_wiar,
}


# ---------------------------------------------------------------------------
# S3 helpers (kept from original)
# ---------------------------------------------------------------------------

def parse_dataset_filter(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_suffix_filter(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    suffixes = set()
    for part in value.split(","):
        suffix = part.strip().lower()
        if not suffix:
            continue
        suffixes.add(suffix if suffix.startswith(".") else f".{suffix}")
    return suffixes or None


def is_s3_uri(value: str | Path) -> bool:
    return str(value).startswith("s3://")


def parse_s3_uri(uri: str | Path) -> tuple[str, str]:
    value = str(uri)
    if not value.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {value}")
    without_scheme = value[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"S3 URI must include a bucket: {value}")
    return bucket, prefix.strip("/")


def make_s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 URIs.") from exc
    endpoint_url = os.getenv("S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL") or "http://207.180.243.242:9000"
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    kwargs = {"endpoint_url": endpoint_url}
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def upload_s3_file(local_path: Path, s3_uri: str | Path) -> dict:
    bucket, key = parse_s3_uri(s3_uri)
    client = make_s3_client()
    client.upload_file(str(local_path), bucket, key)
    return {"bucket": bucket, "key": key, "bytes": local_path.stat().st_size, "uri": f"s3://{bucket}/{key}"}


def upload_s3_directory(local_dir: Path, s3_uri: str | Path) -> dict:
    bucket, prefix = parse_s3_uri(s3_uri)
    prefix = prefix.strip("/")
    client = make_s3_client()
    object_count, total_bytes = 0, 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_key = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{relative_key}" if prefix else relative_key
        client.upload_file(str(path), bucket, key)
        object_count += 1
        total_bytes += path.stat().st_size
    return {"bucket": bucket, "prefix": prefix, "object_count": object_count, "total_bytes": total_bytes, "uri": f"s3://{bucket}/{prefix}"}


def download_s3_prefix(s3_uri: str | Path, destination: Path, *, suffixes: set[str] | None = None) -> dict:
    bucket, prefix = parse_s3_uri(s3_uri)
    client = make_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    page_kwargs = {"Bucket": bucket}
    if prefix:
        page_kwargs["Prefix"] = f"{prefix}/"
    object_count, skipped_count, total_bytes = 0, 0, 0
    for page in paginator.paginate(**page_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if suffixes is not None and Path(key).suffix.lower() not in suffixes:
                skipped_count += 1
                continue
            relative_key = key[len(prefix):].lstrip("/") if prefix else key
            if not relative_key:
                continue
            local_path = destination / relative_key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))
            object_count += 1
            total_bytes += int(obj.get("Size", 0))
    if object_count == 0:
        raise RuntimeError(f"No objects found under {s3_uri}")
    return {"bucket": bucket, "prefix": prefix, "object_count": object_count, "skipped_count": skipped_count, "total_bytes": total_bytes}


@contextmanager
def materialized_bronze_root(bronze_root: str | Path) -> Iterator[tuple[Path, dict]]:
    if not is_s3_uri(bronze_root):
        log.info("Using local bronze root=%s", bronze_root)
        yield Path(bronze_root), {"source_type": "local", "source_uri": str(bronze_root)}
        return
    with tempfile.TemporaryDirectory(prefix="rfpose-bronze-") as tmpdir:
        local_root = Path(tmpdir)
        suffixes = parse_suffix_filter(os.getenv("RFPOSE_S3_STAGE_EXTENSIONS", ".json,.mat,.npy,.csv,.dat"))
        s3_report = download_s3_prefix(bronze_root, local_root, suffixes=suffixes)
        yield local_root, {"source_type": "s3", "source_uri": str(bronze_root), **s3_report}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def bronze_to_silver(
    bronze_root: str | Path,
    silver_out: str | Path,
    *,
    datasets: set[str] | None = None,
    max_samples_per_dataset: int | None = None,
    force: bool = False,
) -> dict:
    if datasets is None:
        datasets = parse_dataset_filter(os.getenv("RFPOSE_BRONZE_DATASETS"))
    if max_samples_per_dataset is None and os.getenv("RFPOSE_MAX_SAMPLES_PER_DATASET"):
        max_samples_per_dataset = int(os.getenv("RFPOSE_MAX_SAMPLES_PER_DATASET"))

    out_dir = Path(silver_out)

    # Idempotent check
    if not force:
        report_path = out_dir / "quality_report.json"
        if report_path.exists():
            try:
                cached = json.loads(report_path.read_text())
                cached["skipped"] = True
                log.info("SKIP bronze_to_silver: output exists at %s (%d samples)", silver_out, cached.get("samples", 0))
                return cached
            except Exception:
                pass

    log.info("START bronze_to_silver: bronze=%s -> silver=%s (datasets=%s, max_samples=%s)",
             bronze_root, silver_out, datasets, max_samples_per_dataset)

    with materialized_bronze_root(bronze_root) as (local_bronze_root, source_report):
        log.info("  Bronze source: %s", source_report.get("source_type", "local"))

        out_dir.mkdir(parents=True, exist_ok=True)
        csi_dir = out_dir / "csi"
        csi_dir.mkdir(exist_ok=True)

        catalog_rows = []
        dataset_counts = Counter()
        quality_issues = []
        t0_total = time.time()

        all_ds_roots = list(existing_dataset_roots(local_bronze_root).items())
        active_datasets = []
        for ds_name, ds_root in all_ds_roots:
            if datasets is not None and ds_name not in datasets:
                log.info("  [%s] Skipped (not in filter)", ds_name)
                continue
            if not ds_root.exists():
                log.info("  [%s] Skipped (not found: %s)", ds_name, ds_root)
                continue
            active_datasets.append((ds_name, ds_root))

        total_ds = len(active_datasets)
        log.info("  Processing %d datasets ...", total_ds)

        for ds_idx, (ds_name, ds_root) in enumerate(active_datasets, 1):
            t0_ds = time.time()
            converter = CONVERTERS[ds_name]
            ds_csi_dir = csi_dir / ds_name
            ds_csi_dir.mkdir(exist_ok=True)

            ds_count = 0
            ds_skipped = 0
            log.info("  [%d/%d] [%s] START root=%s", ds_idx, total_ds, ds_name, ds_root)

            for row, csi_tensor in converter(ds_root, max_samples=max_samples_per_dataset):
                # Validate CSI
                if csi_tensor.size == 0 or np.isnan(csi_tensor).all():
                    ds_skipped += 1
                    quality_issues.append(f"{ds_name}/{row['sample_id']}: empty or all-NaN CSI")
                    continue

                # Save CSI as .npy
                sid = _safe_sample_id(row["sample_id"])
                csi_path = ds_csi_dir / f"{sid}.npy"
                np.save(csi_path, csi_tensor)

                # Add namespace keys
                row["subject_key"] = namespace_key(ds_name, row.get("subject"))
                row["environment_key"] = namespace_key(ds_name, row.get("environment"))
                row["location_key"] = namespace_key(ds_name, row.get("location"))
                row["csi_path"] = str(csi_path.relative_to(out_dir))
                row["csi_shape"] = json.dumps(list(csi_tensor.shape))

                # Serialize pose as JSON string for parquet
                if row.get("pose") is not None:
                    row["pose"] = json.dumps(row["pose"])

                catalog_rows.append(row)
                ds_count += 1
                dataset_counts[ds_name] += 1

                if ds_count % 500 == 0:
                    log.info("    [%s] %d samples saved (%.1fs) ...",
                             ds_name, ds_count, time.time() - t0_ds)

            elapsed_ds = time.time() - t0_ds
            log.info("  [%d/%d %.0f%%] [%s] DONE: %d samples, %d skipped in %.1fs (%.0f samples/s)",
                     ds_idx, total_ds, ds_idx / total_ds * 100,
                     ds_name, ds_count, ds_skipped, elapsed_ds,
                     ds_count / max(elapsed_ds, 0.01))

        elapsed_total = time.time() - t0_total
        total_samples = len(catalog_rows)
        log.info("  Total: %d samples from %d datasets in %.1fs", total_samples, total_ds, elapsed_total)

    # Write catalog parquet
    if pl is not None and catalog_rows:
        catalog_path = out_dir / "catalog.parquet"
        df = pl.DataFrame(catalog_rows)
        df.write_parquet(catalog_path)
        log.info("  Wrote catalog: %s (%d rows, %.1f MB)",
                 catalog_path, len(df), catalog_path.stat().st_size / 1024 / 1024)
    elif catalog_rows:
        catalog_path = out_dir / "catalog.jsonl"
        with open(catalog_path, "w") as f:
            for row in catalog_rows:
                f.write(json.dumps(row) + "\n")
        log.info("  Wrote catalog: %s (%d rows)", catalog_path, len(catalog_rows))

    # Quality report
    report = {
        "samples": total_samples,
        "datasets": dict(sorted(dataset_counts.items())),
        "quality_issues": quality_issues[:100],
        "quality_issues_total": len(quality_issues),
        "status": "ok" if total_samples > 0 else "empty",
        "schema_version": "silver_csi_v2",
        "silver_dir": str(out_dir),
    }
    report["bronze_source"] = source_report
    report["skipped"] = False
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2))
    log.info("DONE bronze_to_silver: %d samples, datasets=%s, status=%s",
             report["samples"], list(report["datasets"].keys()), report["status"])
    return report


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("RFPOSE_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze-root", required=True)
    parser.add_argument("--silver-out", required=True, help="Silver OUTPUT DIRECTORY (not file)")
    parser.add_argument("--datasets", default=os.getenv("RFPOSE_BRONZE_DATASETS"))
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(bronze_to_silver(
        args.bronze_root, args.silver_out,
        datasets=parse_dataset_filter(args.datasets),
        max_samples_per_dataset=args.max_samples_per_dataset,
    ), indent=2))
