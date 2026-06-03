from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator
import argparse
import json
import logging
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

SILVER_COLUMNS = {
    "dataset": None,
    "deployment_id": None,
    "sample_id": None,
    "split": None,
    "source_file": None,
    "buffer_id": None,
    "received_at_ms": None,
    "frame_id": None,
    "timestamp_us": None,
    "node_id": None,
    "node_key": None,
    "tx": None,
    "rx": None,
    "antenna": None,
    "seq": None,
    "rssi": None,
    "noise_floor": None,
    "channel": None,
    "n_subcarriers": None,
    "firmware_version": None,
    "amplitude": None,
    "phase": None,
    "crc32": None,
    "activity": None,
    "activity_id": None,
    "subject": None,
    "subject_key": None,
    "environment": None,
    "environment_key": None,
    "location": None,
    "location_key": None,
    "num_users": None,
    "pose_path": None,
    "pose_index": None,
    "pose": None,
    "pose_joints": None,
    "pose_dim": None,
    "has_pose": False,
    "metadata_json": None,
}

COMMON_POSE_JOINTS = [
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

# MMFi is treated as COCO-17 style:
# nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
MMFI_TO_COMMON = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Wi-Pose is AlphaPose/OpenPose-like:
# nose, neck, right body side, left body side, eyes, ears.
WIPOSE_TO_COMMON = [0, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

# Person-in-WiFi-3D is assumed:
# head, neck, right body side, left body side.
PERSON_WIFI_TO_COMMON = [0, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

ACTIVITY_ALIASES = {
    "sitdown": "sit_down",
    "sit_down": "sit_down",
    "standup": "stand_up",
    "stand_up": "stand_up",
    "pick_up": "pick_up",
    "picking_up_things": "pick_up",
    "jump": "jump",
    "jumping_up": "jump",
    "throw": "throw",
    "throwing_left_side": "throw",
    "throwing_right_side": "throw",
    "high_throw": "throw",
    "wave": "wave",
    "waving_hand_left": "wave",
    "waving_hand_right": "wave",
    "horizontal_arm_wave": "wave",
    "high_arm_wave": "wave",
    "two_hands_wave": "wave",
    "forward_kick": "kick",
    "side_kick": "kick",
    "kicking_left_side": "kick",
    "kicking_right_side": "kick",
    "squat": "squat",
    "crouch": "squat",
    "bend": "bend",
    "bowing": "bend",
    "walk": "walk",
    "run": "run",
    "fall": "fall",
    "lie_down": "lie_down",
}


def namespace_key(dataset: str | None, value) -> str | None:
    if dataset is None or value is None:
        return None
    return f"{dataset}:{value}"


def stream_key(dataset: str | None, sample_id, node_id) -> str | None:
    if dataset is None or sample_id is None or node_id is None:
        return None
    return f"{dataset}:{sample_id}:node:{node_id}"


def normalize_activity(activity) -> str | None:
    if activity is None:
        return None

    normalized = str(activity).strip().lower().replace(" ", "_").replace("-", "_")
    return ACTIVITY_ALIASES.get(normalized, normalized)


def make_silver_row(**values) -> dict:
    row = dict(SILVER_COLUMNS)
    row.update(values)

    row["activity"] = normalize_activity(row.get("activity"))

    dataset = row.get("dataset")
    row["node_key"] = row.get("node_key") or stream_key(
        dataset, row.get("sample_id"), row.get("node_id")
    )
    row["subject_key"] = row.get("subject_key") or namespace_key(
        dataset, row.get("subject")
    )
    row["environment_key"] = row.get("environment_key") or namespace_key(
        dataset, row.get("environment")
    )
    row["location_key"] = row.get("location_key") or namespace_key(
        dataset, row.get("location")
    )

    return row


def to_float_list(values) -> list[float]:
    return np.asarray(values, dtype=np.float32).tolist()


def select_common_pose_joints(pose, indices: list[int], *, dataset: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)

    if pose.ndim != 2:
        raise ValueError(f"{dataset} pose must be 2D [joints, dims], got {pose.shape}")

    if pose.shape[0] <= max(indices):
        raise ValueError(
            f"{dataset} pose has {pose.shape[0]} joints, "
            f"but common mapping requires index {max(indices)}"
        )

    return pose[indices]


def pose_fields(common_pose: np.ndarray) -> dict:
    return {
        "pose": common_pose.astype(np.float32).tolist(),
        "pose_joints": COMMON_POSE_JOINTS,
        "pose_dim": int(common_pose.shape[1]),
        "has_pose": True,
    }


def maybe_json(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def limit_samples(samples: Iterable[dict], max_samples: int | None) -> Iterable[dict]:
    if max_samples is None:
        yield from samples
        return

    for idx, sample in enumerate(samples):
        if idx >= max_samples:
            break
        yield sample


def iter_bronze_batches(bronze_root: str | Path) -> Iterable[Path]:
    root = Path(bronze_root)
    yield from sorted(root.rglob("*.json"))


def decode_packet_record(packet: dict, deployment_id: str, source_file: str) -> dict:
    pkt = (
        json.loads(packet["packet_json"])
        if isinstance(packet.get("packet_json"), str)
        else packet
    )
    amp = pkt.get("amplitude") or []
    return make_silver_row(
        dataset="self_captured",
        deployment_id=deployment_id,
        source_file=source_file,
        buffer_id=packet.get("id"),
        received_at_ms=packet.get("received_at_ms"),
        node_id=int(pkt.get("node_id", packet.get("node_id", 0))),
        seq=int(pkt.get("seq", packet.get("seq", 0))),
        timestamp_us=int(pkt.get("timestamp_us", packet.get("timestamp_us", 0))),
        rssi=int(pkt.get("rssi", 0)),
        noise_floor=int(pkt.get("noise_floor", 0)),
        channel=int(pkt.get("channel", 0)),
        n_subcarriers=int(pkt.get("n_subcarriers", len(amp))),
        firmware_version=int(pkt.get("firmware_version", 0)),
        amplitude=amp,
        crc32=int(pkt.get("crc32", 0)),
    )


def iter_json_packet_rows(bronze_root: str | Path) -> Iterator[dict]:
    for file in iter_bronze_batches(bronze_root):
        obj = json.loads(file.read_text())
        deployment_id = obj.get("deployment_id", "unknown")
        for packet in obj.get("packets", []):
            yield decode_packet_record(packet, deployment_id, str(file))


def iter_wimans_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:
    samples = iter_wimans_samples(
        root,
        single_person_only=False,
        include_empty_room=True,
    )

    for sample in limit_samples(samples, max_samples):
        amp = load_wimans_amp(sample["amp_path"])
        if amp.ndim != 4:
            raise ValueError(f"Unexpected WiMANS amp shape: {amp.shape}")

        time_steps, tx_count, rx_count, n_subcarriers = amp.shape
        activity = sample["activities"][0] if sample["activities"] else None
        location = sample["locations"][0] if sample["locations"] else None

        for t in range(time_steps):
            for tx in range(tx_count):
                for rx in range(rx_count):
                    yield make_silver_row(
                        dataset="wimans",
                        sample_id=sample["label"],
                        source_file=str(sample["amp_path"]),
                        timestamp_us=t,
                        seq=t,
                        node_id=tx * rx_count + rx + 1,
                        tx=tx + 1,
                        rx=rx + 1,
                        n_subcarriers=n_subcarriers,
                        amplitude=to_float_list(amp[t, tx, rx, :]),
                        activity=activity,
                        environment=sample["environment"],
                        location=location,
                        num_users=sample["num_users"],
                        has_pose=False,
                        metadata_json=maybe_json(
                            {
                                "wifi_band": sample["wifi_band"],
                                "activities": sample["activities"],
                                "locations": sample["locations"],
                                "mat_path": str(sample["mat_path"])
                                if sample["mat_path"]
                                else None,
                                "video_path": str(sample["video_path"])
                                if sample["video_path"]
                                else None,
                            }
                        ),
                    )


def iter_person_wifi_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:
    for split in ("train", "test"):
        split_dir = root / f"{split}_data"
        if not split_dir.exists():
            continue

        samples = index_person_wifi_samples(
            root,
            split=split,
            single_person_only=False,
        )

        for sample in limit_samples(samples, max_samples):
            csi = load_person_wifi_csi_mat(sample["csi_path"])
            keypoint = load_person_wifi_keypoint_npy(
                sample["keypoint_path"],
                single_person_only=False,
            )
            common_pose = select_common_pose_joints(
                keypoint[0],
                PERSON_WIFI_TO_COMMON,
                dataset="person_in_wifi_3d",
            )
            amp = np.abs(csi).astype(np.float32)
            phase = np.angle(csi).astype(np.float32)
            ant1_count, ant2_count, n_subcarriers, time_steps = csi.shape

            for t in range(time_steps):
                for ant1 in range(ant1_count):
                    for ant2 in range(ant2_count):
                        yield make_silver_row(
                            dataset="person_in_wifi_3d",
                            sample_id=sample["name"],
                            split=split,
                            source_file=str(sample["csi_path"]),
                            timestamp_us=t,
                            seq=t,
                            node_id=ant1 * ant2_count + ant2 + 1,
                            tx=ant1 + 1,
                            rx=ant2 + 1,
                            n_subcarriers=n_subcarriers,
                            amplitude=to_float_list(amp[ant1, ant2, :, t]),
                            phase=to_float_list(phase[ant1, ant2, :, t]),
                            num_users=sample["num_people"],
                            pose_path=str(sample["keypoint_path"]),
                            **pose_fields(common_pose),
                        )


def iter_wipose_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:
    samples = index_wipose_samples(root)

    for sample in limit_samples(samples, max_samples):
        csi, pose = load_wipose_mat(sample["path"])
        common_pose = select_common_pose_joints(
            pose,
            WIPOSE_TO_COMMON,
            dataset="wipose",
        )
        tx_count, rx_count, n_subcarriers, time_steps, _channels = csi.shape

        for t in range(time_steps):
            for tx in range(tx_count):
                for rx in range(rx_count):
                    yield make_silver_row(
                        dataset="wipose",
                        sample_id=sample["path"].stem,
                        split=sample["split"],
                        source_file=str(sample["path"]),
                        frame_id=sample["frame"],
                        timestamp_us=t,
                        seq=t,
                        node_id=tx * rx_count + rx + 1,
                        tx=tx + 1,
                        rx=rx + 1,
                        n_subcarriers=n_subcarriers,
                        amplitude=to_float_list(csi[tx, rx, :, t, 0]),
                        activity=sample["action"],
                        subject=sample["subject"],
                        **pose_fields(common_pose),
                    )


def iter_mmfi_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:
    samples = index_mmfi_wifi_csi_samples(root, require_pose=False)

    for sample in limit_samples(samples, max_samples):
        amp, phase = load_mmfi_csi_mat(sample["path"])
        antenna_count, n_subcarriers, time_steps = amp.shape
        pose_index = (
            mmfi_frame_to_index(sample["frame"])
            if sample["pose_path"] is not None
            else None
        )
        common_pose = None
        if sample["pose_path"] is not None:
            poses = load_mmfi_pose_npy(sample["pose_path"])
            common_pose = select_common_pose_joints(
                poses[pose_index],
                MMFI_TO_COMMON,
                dataset="mmfi",
            )

        for t in range(time_steps):
            for antenna in range(antenna_count):
                pose_values = (
                    pose_fields(common_pose)
                    if common_pose is not None
                    else {"has_pose": False}
                )
                yield make_silver_row(
                    dataset="mmfi",
                    sample_id=str(sample["path"].parent.parent),
                    source_file=str(sample["path"]),
                    frame_id=sample["frame"],
                    timestamp_us=t,
                    seq=t,
                    node_id=antenna + 1,
                    rx=antenna + 1,
                    antenna=antenna + 1,
                    n_subcarriers=n_subcarriers,
                    amplitude=to_float_list(amp[antenna, :, t]),
                    phase=to_float_list(phase[antenna, :, t]),
                    activity=sample["action"],
                    subject=sample["subject"],
                    environment=sample["environment"],
                    pose_path=str(sample["pose_path"]) if sample["pose_path"] else None,
                    pose_index=pose_index,
                    **pose_values,
                )


def iter_uthar_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:
    for split in ("train", "val", "test"):
        x, y = load_uthar_arrays(root, split)
        sample_count = len(y) if max_samples is None else min(len(y), max_samples)

        for sample_idx in range(sample_count):
            sample = x[sample_idx].reshape(250, 3, 30)
            label = int(y[sample_idx])

            for t in range(sample.shape[0]):
                for antenna in range(sample.shape[1]):
                    yield make_silver_row(
                        dataset="uthar",
                        sample_id=f"{split}_{sample_idx}",
                        split=split,
                        source_file=str(root / "data" / f"X_{split}.csv"),
                        timestamp_us=t,
                        seq=t,
                        node_id=antenna + 1,
                        antenna=antenna + 1,
                        n_subcarriers=30,
                        amplitude=to_float_list(sample[t, antenna, :]),
                        activity=UT_HAR_ACTIONS.get(label, f"unknown_{label}"),
                        activity_id=label,
                        has_pose=False,
                    )


def iter_wiar_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:
    samples = index_wiar_samples(root)

    for sample in limit_samples(samples, max_samples):
        loaded = load_wiar_sample(sample["path"])
        csi = loaded["csi"]
        time_steps, n_subcarriers, rx_count, tx_count = csi.shape
        amp = np.abs(csi).astype(np.float32)
        phase = np.angle(csi).astype(np.float32)

        for t in range(time_steps):
            for rx in range(rx_count):
                for tx in range(tx_count):
                    yield make_silver_row(
                        dataset="wiar",
                        sample_id=f"{sample['volunteer']}_{sample['activity_id']}_{sample['sample_id']}",
                        source_file=str(sample["path"]),
                        timestamp_us=t,
                        seq=t,
                        node_id=tx * rx_count + rx + 1,
                        tx=tx + 1,
                        rx=rx + 1,
                        n_subcarriers=n_subcarriers,
                        amplitude=to_float_list(amp[t, :, rx, tx]),
                        phase=to_float_list(phase[t, :, rx, tx]),
                        activity=sample["activity_name"],
                        activity_id=sample["activity_id"],
                        subject=sample["volunteer"],
                        has_pose=False,
                    )


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


def iter_public_dataset_rows(
    bronze_root: str | Path,
    *,
    datasets: set[str] | None = None,
    max_samples_per_dataset: int | None = None,
) -> Iterator[dict]:
    converters = {
        "wimans": iter_wimans_rows,
        "person_in_wifi_3d": iter_person_wifi_rows,
        "wipose": iter_wipose_rows,
        "mmfi": iter_mmfi_rows,
        "uthar": iter_uthar_rows,
        "wiar": iter_wiar_rows,
    }

    all_datasets = list(existing_dataset_roots(bronze_root).items())
    total_datasets = len(all_datasets)
    active_idx = 0
    for dataset_name, dataset_root in all_datasets:
        if datasets is not None and dataset_name not in datasets:
            log.info("Skipping dataset=%s because it is not selected", dataset_name)
            continue

        if not dataset_root.exists():
            log.info("Skipping dataset=%s because root is missing: %s", dataset_name, dataset_root)
            continue

        log.info(
            "Starting bronze-to-silver dataset=%s root=%s max_samples_per_dataset=%s",
            dataset_name,
            dataset_root,
            max_samples_per_dataset if max_samples_per_dataset is not None else "all",
        )
        row_count = 0
        for row in converters[dataset_name](
            dataset_root,
            max_samples=max_samples_per_dataset,
        ):
            row_count += 1
            yield row
        log.info("Finished bronze-to-silver dataset=%s rows=%d", dataset_name, row_count)


def iter_silver_rows(
    bronze_root: str | Path,
    *,
    datasets: set[str] | None = None,
    max_samples_per_dataset: int | None = None,
) -> Iterator[dict]:
    if datasets is None or "self_captured" in datasets:
        log.info("Starting bronze-to-silver dataset=self_captured root=%s", bronze_root)
        row_count = 0
        for row in iter_json_packet_rows(bronze_root):
            row_count += 1
            yield row
        log.info("Finished bronze-to-silver dataset=self_captured rows=%d", row_count)
    else:
        log.info("Skipping dataset=self_captured because it is not selected")

    yield from iter_public_dataset_rows(
        bronze_root,
        datasets=datasets,
        max_samples_per_dataset=max_samples_per_dataset,
    )


def write_rows(rows: list[dict], silver_out: str | Path) -> None:
    if is_s3_uri(silver_out):
        with tempfile.TemporaryDirectory(prefix="rfpose-silver-") as tmpdir:
            local_out = Path(tmpdir) / Path(str(silver_out)).name
            log.info(
                "Writing silver rows to temporary file before S3 upload path=%s rows=%d",
                local_out,
                len(rows),
            )
            write_rows(rows, local_out)
            upload_report = upload_s3_file(local_out, silver_out)
            log.info(
                "Uploaded silver rows to %s bytes=%d",
                upload_report["uri"],
                upload_report["bytes"],
            )
        return

    out = Path(silver_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if pl is not None and out.suffix == ".parquet":
        pl.DataFrame(rows).write_parquet(out)
        log.info("Wrote silver parquet path=%s rows=%d bytes=%d", out, len(rows), out.stat().st_size)
        return

    out.write_text("\n".join(json.dumps(row) for row in rows))
    log.info("Wrote silver jsonl path=%s rows=%d bytes=%d", out, len(rows), out.stat().st_size)


def build_quality_report(rows: list[dict]) -> dict:
    node_ids = sorted({row["node_id"] for row in rows if row.get("node_id") is not None})
    datasets = Counter(row["dataset"] for row in rows)
    seq_drops = 0
    seqs_by_stream = defaultdict(list)

    for row in rows:
        if row.get("seq") is None or row.get("node_id") is None:
            continue
        key = (row.get("dataset"), row.get("sample_id"), row.get("node_id"))
        seqs_by_stream[key].append(int(row["seq"]))

    for seqs in seqs_by_stream.values():
        seqs = sorted(set(seqs))
        for a, b in zip(seqs, seqs[1:]):
            if b > a + 1:
                seq_drops += b - a - 1

    return {
        "rows": len(rows),
        "datasets": dict(sorted(datasets.items())),
        "node_ids": node_ids,
        "node_count": len(node_ids),
        "seq_drops_est": seq_drops,
        "status": "ok" if rows else "empty",
        "schema_version": "silver_csi_v1",
    }


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

    without_scheme = value[len("s3://") :]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"S3 URI must include a bucket: {value}")
    return bucket, prefix.strip("/")


def make_s3_client():
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "boto3 is required to read bronze data from s3:// URIs."
        ) from exc

    endpoint_url = (
        os.getenv("S3_ENDPOINT_URL")
        or os.getenv("AWS_ENDPOINT_URL")
        or "http://207.180.243.242:9000"
    )
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD")

    kwargs = {"endpoint_url": endpoint_url}
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key

    return boto3.client("s3", **kwargs)


def download_s3_prefix(
    s3_uri: str | Path,
    destination: Path,
    *,
    suffixes: set[str] | None = None,
) -> dict:
    bucket, prefix = parse_s3_uri(s3_uri)
    log.info(
        "Starting S3 bronze staging uri=%s bucket=%s prefix=%s suffixes=%s destination=%s",
        s3_uri,
        bucket,
        prefix,
        sorted(suffixes) if suffixes is not None else "all",
        destination,
    )
    client = make_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    page_kwargs = {"Bucket": bucket}
    if prefix:
        page_kwargs["Prefix"] = f"{prefix}/"

    object_count = 0
    skipped_count = 0
    total_bytes = 0
    latest_key = None

    for page in paginator.paginate(**page_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if suffixes is not None and Path(key).suffix.lower() not in suffixes:
                skipped_count += 1
                continue

            relative_key = key[len(prefix) :].lstrip("/") if prefix else key
            if not relative_key:
                continue

            local_path = destination / relative_key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))

            object_count += 1
            total_bytes += int(obj.get("Size", 0))
            latest_key = key
            if object_count % 1000 == 0:
                log.info(
                    "S3 bronze staging progress downloaded=%d skipped=%d bytes=%d latest_key=%s",
                    object_count,
                    skipped_count,
                    total_bytes,
                    latest_key,
                )

    if object_count == 0:
        raise RuntimeError(f"No objects found under {s3_uri}")

    log.info(
        "Finished S3 bronze staging uri=%s downloaded=%d skipped=%d bytes=%d latest_key=%s",
        s3_uri,
        object_count,
        skipped_count,
        total_bytes,
        latest_key,
    )
    return {
        "bucket": bucket,
        "prefix": prefix,
        "object_count": object_count,
        "skipped_count": skipped_count,
        "total_bytes": total_bytes,
        "latest_key": latest_key,
    }


def download_s3_file(s3_uri: str | Path, destination: Path) -> dict:
    bucket, key = parse_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"S3 input URI must include an object key: {s3_uri}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading S3 file uri=%s destination=%s", s3_uri, destination)
    client = make_s3_client()
    client.download_file(bucket, key, str(destination))
    log.info("Downloaded S3 file uri=%s bytes=%d", s3_uri, destination.stat().st_size)
    return {
        "bucket": bucket,
        "key": key,
        "bytes": destination.stat().st_size,
        "uri": f"s3://{bucket}/{key}",
    }


def upload_s3_file(local_path: Path, s3_uri: str | Path) -> dict:
    bucket, key = parse_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"S3 output URI must include an object key: {s3_uri}")

    client = make_s3_client()
    log.info("Uploading file to S3 local_path=%s uri=%s bytes=%d", local_path, s3_uri, local_path.stat().st_size)
    client.upload_file(str(local_path), bucket, key)
    log.info("Uploaded file to S3 uri=s3://%s/%s bytes=%d", bucket, key, local_path.stat().st_size)
    return {
        "bucket": bucket,
        "key": key,
        "bytes": local_path.stat().st_size,
        "uri": f"s3://{bucket}/{key}",
    }


def upload_s3_directory(local_dir: Path, s3_uri: str | Path) -> dict:
    bucket, prefix = parse_s3_uri(s3_uri)
    prefix = prefix.strip("/")
    client = make_s3_client()
    log.info("Uploading directory to S3 local_dir=%s uri=%s", local_dir, s3_uri)

    object_count = 0
    total_bytes = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue

        relative_key = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{relative_key}" if prefix else relative_key
        client.upload_file(str(path), bucket, key)
        object_count += 1
        total_bytes += path.stat().st_size

    log.info(
        "Uploaded directory to S3 uri=%s object_count=%d bytes=%d",
        s3_uri,
        object_count,
        total_bytes,
    )
    return {
        "bucket": bucket,
        "prefix": prefix,
        "object_count": object_count,
        "total_bytes": total_bytes,
        "uri": f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}",
    }


@contextmanager
def materialized_bronze_root(bronze_root: str | Path) -> Iterator[tuple[Path, dict]]:
    if not is_s3_uri(bronze_root):
        log.info("Using local bronze root=%s", bronze_root)
        yield Path(bronze_root), {"source_type": "local", "source_uri": str(bronze_root)}
        return

    with tempfile.TemporaryDirectory(prefix="rfpose-bronze-") as tmpdir:
        local_root = Path(tmpdir)
        suffixes = parse_suffix_filter(
            os.getenv("RFPOSE_S3_STAGE_EXTENSIONS", ".json,.mat,.npy,.csv,.dat")
        )
        log.info("Materializing S3 bronze root=%s stage_extensions=%s", bronze_root, sorted(suffixes) if suffixes is not None else "all")
        s3_report = download_s3_prefix(
            bronze_root,
            local_root,
            suffixes=suffixes,
        )
        yield local_root, {
            "source_type": "s3",
            "source_uri": str(bronze_root),
            "staged_root": str(local_root),
            "stage_extensions": sorted(suffixes) if suffixes is not None else "all",
            **s3_report,
        }


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

    # --- Idempotent check ---
    if not force and not is_s3_uri(silver_out):
        out_path = Path(silver_out)
        report_path = out_path.parent / "quality_report.json"
        if out_path.exists() and report_path.exists():
            try:
                cached = json.loads(report_path.read_text())
                cached["skipped"] = True
                log.info("SKIP bronze_to_silver: output exists at %s (%d rows)", silver_out, cached.get("rows", 0))
                return cached
            except Exception:
                pass
    elif not force and is_s3_uri(silver_out):
        try:
            bucket, key = parse_s3_uri(silver_out)
            client = make_s3_client()
            client.head_object(Bucket=bucket, Key=key)
            parent_prefix = key.rsplit("/", 1)[0] if "/" in key else ""
            report_key = f"{parent_prefix}/quality_report.json" if parent_prefix else "quality_report.json"
            resp = client.get_object(Bucket=bucket, Key=report_key)
            cached = json.loads(resp["Body"].read().decode())
            cached["skipped"] = True
            log.info("SKIP bronze_to_silver: S3 output exists at %s (%d rows)", silver_out, cached.get("rows", 0))
            return cached
        except Exception:
            pass

    log.info("START bronze_to_silver: bronze=%s -> silver=%s (datasets=%s)", bronze_root, silver_out, datasets)

    with materialized_bronze_root(bronze_root) as (local_bronze_root, source_report):
        log.info("  Bronze source: %s", source_report.get("source_type", "local"))
        rows = list(
            iter_silver_rows(
                local_bronze_root,
                datasets=datasets,
                max_samples_per_dataset=max_samples_per_dataset,
            )
        )

    report = build_quality_report(rows)
    report["bronze_source"] = source_report
    report["skipped"] = False
    log.info("DONE bronze_to_silver: %d rows, datasets=%s, status=%s", report["rows"], list(report.get("datasets", {}).keys()), report["status"])

    if is_s3_uri(silver_out):
        with tempfile.TemporaryDirectory(prefix="rfpose-silver-report-") as tmpdir:
            report_path = Path(tmpdir) / "quality_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            bucket, key = parse_s3_uri(silver_out)
            parent_prefix = key.rsplit("/", 1)[0] if "/" in key else ""
            report_key = (
                f"{parent_prefix}/quality_report.json"
                if parent_prefix
                else "quality_report.json"
            )
            report_uri = f"s3://{bucket}/{report_key}"
            log.info("Uploading silver quality report uri=%s", report_uri)
            upload_s3_file(report_path, report_uri)
        write_rows(rows, silver_out)
        log.info("Finished bronze_to_silver silver_out=%s rows=%d", silver_out, len(rows))
        return report

    write_rows(rows, silver_out)
    out = Path(silver_out)
    (out.parent / "quality_report.json").write_text(json.dumps(report, indent=2))
    log.info("Wrote silver quality report path=%s", out.parent / "quality_report.json")
    log.info("Finished bronze_to_silver silver_out=%s rows=%d", silver_out, len(rows))
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("RFPOSE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze-root", required=True)
    parser.add_argument("--silver-out", required=True)
    parser.add_argument(
        "--datasets",
        default=os.getenv("RFPOSE_BRONZE_DATASETS"),
        help=(
            "Comma-separated dataset filter. Valid values include self_captured, "
            "wimans, person_in_wifi_3d, wipose, mmfi, uthar, wiar."
        ),
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=(
            int(os.getenv("RFPOSE_MAX_SAMPLES_PER_DATASET"))
            if os.getenv("RFPOSE_MAX_SAMPLES_PER_DATASET")
            else None
        ),
    )
    args = parser.parse_args()

    print(
        json.dumps(
            bronze_to_silver(
                args.bronze_root,
                args.silver_out,
                datasets=parse_dataset_filter(args.datasets),
                max_samples_per_dataset=args.max_samples_per_dataset,
            ),
            indent=2,
        )
    )
