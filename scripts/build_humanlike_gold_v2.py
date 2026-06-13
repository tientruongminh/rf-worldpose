#!/usr/bin/env python3
"""Build human-like pose Gold v2 datasets for MM-Fi and WiPose.

This intentionally targets good-looking, trainable skeleton output rather than
paper-exact reproduction. WiPose bronze SkeletonPoints are 2D keypoints laid out
as [x1..x18, y1..y18, confidence1..confidence18], so this builder creates a
pseudo-3D EDA-style skeleton. MM-Fi uses the existing H36M 17-joint 3D gold when
raw bronze is not staged on the compute filesystem, sanitizing CSI and rebuilding
sequence-level splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover - reported at runtime for WiPose only
    h5py = None


LOG = logging.getLogger("humanlike_gold_v2")

WINDOW = 60
STRIDE = 21
WIPOSE_N_SUB = 1350
MMFI_N_SUB = 342

ACTION_LABELS = {
    "unlabeled": 0,
    "walk": 1,
    "run": 2,
    "sit_down": 3,
    "stand_up": 4,
    "fall": 5,
    "lie_down": 6,
    "jump": 7,
    "squat": 8,
    "bend": 9,
    "hand_clap": 10,
    "wave": 11,
    "phone_call": 12,
    "drink_water": 13,
    "throw": 14,
    "pick_up": 15,
    "push": 16,
    "pull": 17,
    "kick": 18,
    "toss_paper": 19,
    "draw_x": 20,
    "draw_tick": 21,
    "rotation": 22,
    "circle": 23,
    "crouch": 24,
    "nothing": 25,
    "empty": 26,
    "other": 27,
}

WIPOSE_JOINTS = [
    "Head",
    "Neck",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Foot",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Foot",
]

H36M17_JOINTS = [
    "Pelvis",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "Spine",
    "Thorax",
    "Nose",
    "Head_Top",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
]

WIPOSE_EDA_BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [1, 5], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13],
    [8, 11], [0, 14], [14, 16], [0, 15], [15, 17],
]

H36M17_BONES = [
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13],
    [8, 14], [14, 15], [15, 16],
]


def stable_split(key: str, val_ratio: float = 0.1, test_ratio: float = 0.1) -> str:
    """Deterministic group split from a sequence/sample id."""
    bucket = sum((i + 1) * ord(c) for i, c in enumerate(key)) % 10_000
    frac = bucket / 10_000.0
    if frac < test_ratio:
        return "test"
    if frac < test_ratio + val_ratio:
        return "val"
    return "train"


_MMFI_SID_RE = re.compile(r"E\d+_S(\d+)_A\d+")


def mmfi_subject_split(sample_id: str) -> str:
    """Cross-subject split for MM-Fi, balanced across all 4 environments.

    MM-Fi ids are ``E{env}_S{subject}_A{action}``. Splitting on the full id only
    avoids *window* leakage — the same subject's other actions still leak across
    splits. Splitting on the subject (balanced per environment so train always
    sees every environment) gives a genuine cross-subject evaluation:
        test : subject last digit in {5, 0}  → 2 subj/env  (8 subjects, ~20%)
        val  : subject last digit == 8        → 1 subj/env  (4 subjects, ~10%)
        train: remaining 28 subjects                        (~70%)
    """
    m = _MMFI_SID_RE.match(sample_id)
    if not m:
        return stable_split(sample_id)
    last = int(m.group(1)) % 10
    if last in (5, 0):
        return "test"
    if last == 8:
        return "val"
    return "train"


def stratified_group_splits(
    group_to_action: dict[str, str],
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """Split whole sequences, stratified by action, to avoid window leakage."""
    action_groups: dict[str, list[str]] = {}
    for group, action in group_to_action.items():
        action_groups.setdefault(action, []).append(group)

    split_by_group: dict[str, str] = {}
    for action, groups in action_groups.items():
        ordered = sorted(
            groups,
            key=lambda g: hashlib.sha1(f"{seed}:{action}:{g}".encode("utf-8")).hexdigest(),
        )
        n = len(ordered)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        n_test = min(n_test, n)
        n_val = min(n_val, max(0, n - n_test))
        for group in ordered[:n_test]:
            split_by_group[group] = "test"
        for group in ordered[n_test:n_test + n_val]:
            split_by_group[group] = "val"
        for group in ordered[n_test + n_val:]:
            split_by_group[group] = "train"
    return split_by_group


def action_from_sample(sample_id: str) -> str:
    token = sample_id.split("_")[0].lower()
    aliases = {
        "clap": "hand_clap",
        "phone": "phone_call",
        "drink": "drink_water",
        "pickup": "pick_up",
    }
    return aliases.get(token, token if token in ACTION_LABELS else "other")


def finite_stats(x_path: Path, chunk: int = 256) -> dict[str, float]:
    arr = np.load(x_path, mmap_mode="r")
    total = 0
    sum_v = 0.0
    sum_sq = 0.0
    mn = math.inf
    mx = -math.inf
    for start in range(0, arr.shape[0], chunk):
        block = np.asarray(arr[start:start + chunk], dtype=np.float64)
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        total += block.size
        sum_v += float(block.sum())
        sum_sq += float((block * block).sum())
        mn = min(mn, float(block.min()))
        mx = max(mx, float(block.max()))
    mean = sum_v / max(total, 1)
    var = max(sum_sq / max(total, 1) - mean * mean, 0.0)
    return {"mean": mean, "std": math.sqrt(var) + 1e-6, "min": mn, "max": mx}


def write_common_files(
    ds_dir: Path,
    *,
    dataset: str,
    skeleton: str,
    joint_names: list[str],
    bones: list[list[int]],
    metadata: list[dict],
    pose: np.ndarray,
    action_label: np.ndarray,
    action_mask: np.ndarray,
    source: str,
) -> None:
    ds_dir.mkdir(parents=True, exist_ok=True)
    splits = [m["split"] for m in metadata]
    n = len(metadata)
    np.savez_compressed(
        ds_dir / "y.npz",
        pose=pose.astype(np.float32),
        pose_mask=np.ones(n, dtype=np.int64),
        action_label=action_label.astype(np.int64),
        action_mask=action_mask.astype(np.int64),
        activity=action_label.astype(np.int64),
        activity_id=action_label.astype(np.int64),
        activity_mask=action_mask.astype(np.int64),
    )
    np.savez_compressed(ds_dir / "metadata.npz", metadata=np.array(metadata, dtype=object))
    stats = finite_stats(ds_dir / "x.npy")
    (ds_dir / "normalization.json").write_text(json.dumps(stats, indent=2))
    manifest = {
        "dataset": dataset,
        "source": source,
        "purpose": "humanlike_pose_v2",
        "num_samples": n,
        "splits": {s: splits.count(s) for s in ("train", "val", "test")},
        "x_shape": list(np.load(ds_dir / "x.npy", mmap_mode="r").shape),
        "pose_shape": list(pose.shape),
        "window_frames": WINDOW,
        "stride": STRIDE,
        "skeleton": skeleton,
        "joint_names": joint_names,
        "bones": bones,
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (ds_dir / "skeleton.json").write_text(
        json.dumps({"name": skeleton, "joint_names": joint_names, "bones": bones}, indent=2)
    )
    (ds_dir / "label_maps.json").write_text(json.dumps({"activity": ACTION_LABELS}, indent=2))
    LOG.info("%s: wrote %d windows; splits=%s", dataset, n, manifest["splits"])


def smooth_and_normalize_wipose(raw_seq: np.ndarray, conf_thr: float = 0.05) -> np.ndarray:
    """Convert raw [T,18,3] x/y/conf into root-relative pseudo-3D."""
    xy = raw_seq[:, :, :2].astype(np.float32).copy()
    conf = raw_seq[:, :, 2].astype(np.float32)
    t = np.arange(len(raw_seq))
    for j in range(xy.shape[1]):
        valid = np.isfinite(xy[:, j]).all(axis=1) & (conf[:, j] > conf_thr)
        for d in range(2):
            if valid.sum() >= 2:
                xy[:, j, d] = np.interp(t, t[valid], xy[valid, j, d])
            elif valid.sum() == 1:
                xy[:, j, d] = xy[valid, j, d][0]
        if len(raw_seq) >= 3:
            xy[1:-1, j] = 0.25 * xy[:-2, j] + 0.5 * xy[1:-1, j] + 0.25 * xy[2:, j]

    heights = np.nanmax(xy[:, :, 1], axis=1) - np.nanmin(xy[:, :, 1], axis=1)
    widths = np.nanmax(xy[:, :, 0], axis=1) - np.nanmin(xy[:, :, 0], axis=1)
    scale = float(np.nanmedian(np.maximum(heights, widths)))
    if not np.isfinite(scale) or scale < 1.0:
        scale = 256.0

    root = xy[:, 1:2, :]  # neck
    centered = (xy - root) / scale
    pseudo = np.zeros((len(raw_seq), 18, 3), dtype=np.float32)
    pseudo[:, :, 0] = centered[:, :, 0]
    pseudo[:, :, 1] = 0.0
    pseudo[:, :, 2] = -centered[:, :, 1]
    np.nan_to_num(pseudo, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return pseudo


def load_wipose_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if h5py is None:
        raise RuntimeError("h5py is required to build WiPose from bronze")
    with h5py.File(path, "r") as f:
        csi = np.asarray(f["CSI"])
        skel = np.asarray(f["SkeletonPoints"], dtype=np.float32).squeeze().reshape(-1)
    amp = np.abs(csi).reshape(-1).astype(np.float32)
    phase = np.angle(csi).reshape(-1).astype(np.float32) if np.iscomplexobj(csi) else np.zeros_like(amp)
    raw_pose = skel.reshape(3, 18).T.astype(np.float32)  # x,y,confidence
    return np.stack([amp, phase], axis=0), raw_pose


def process_wipose_sequence(args: tuple[str, str, list[str]]) -> dict | None:
    split_name, sample_id, files_s = args
    files = [Path(p) for p in files_s]
    csi_frames = []
    raw_pose = []
    for p in files:
        try:
            csi, pose = load_wipose_frame(p)
        except Exception as exc:
            LOG.warning("skip frame %s: %s", p, exc)
            continue
        if csi.shape[-1] != WIPOSE_N_SUB:
            csi = csi[:, :WIPOSE_N_SUB]
        csi_frames.append(csi)
        raw_pose.append(pose)
    if len(csi_frames) < WINDOW:
        return None
    csi_arr = np.stack(csi_frames).astype(np.float32)  # [T,2,1350]
    pose_arr = smooth_and_normalize_wipose(np.stack(raw_pose))
    action = action_from_sample(sample_id)
    action_id = ACTION_LABELS.get(action, ACTION_LABELS["other"])
    seq_split = "test" if split_name.lower() == "test" else stable_split(sample_id, test_ratio=0.0)
    xs, ys, metas, labels = [], [], [], []
    for start in range(0, len(csi_arr) - WINDOW + 1, STRIDE):
        end = start + WINDOW
        xs.append(csi_arr[start:end].transpose(1, 0, 2))
        ys.append(pose_arr[start:end])
        metas.append({
            "dataset": "wipose",
            "sample_id": sample_id,
            "source_split": split_name,
            "split": seq_split,
            "window_start": start,
            "action": action,
        })
        labels.append(action_id)
    return {"x": np.stack(xs), "pose": np.stack(ys), "metadata": metas, "action_label": np.array(labels)}


def find_wipose_sequences(bronze_root: Path) -> list[tuple[str, str, list[str]]]:
    base = bronze_root
    candidates = [
        bronze_root / "wipose" / "raw_mat" / "Wi-Pose",
        bronze_root / "public" / "wipose" / "raw_mat" / "Wi-Pose",
        bronze_root / "Wi-Pose",
        bronze_root,
    ]
    for c in candidates:
        if (c / "Train").exists() or (c / "Test").exists():
            base = c
            break
    sequences: list[tuple[str, str, list[str]]] = []
    for split in ("Train", "Test"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for p in sorted(split_dir.glob("*.mat")):
            stem = p.stem
            if "-frame" not in stem:
                continue
            sample_id, frame_s = stem.rsplit("-frame", 1)
            try:
                frame = int(frame_s)
            except ValueError:
                continue
            grouped.setdefault(sample_id, []).append((frame, p))
        for sample_id, frames in grouped.items():
            frames.sort(key=lambda x: x[0])
            sequences.append((split, sample_id, [str(p) for _, p in frames]))
    return sequences


def build_wipose(bronze_root: Path, out_root: Path, workers: int) -> None:
    sequences = find_wipose_sequences(bronze_root)
    if not sequences:
        raise FileNotFoundError(f"No WiPose sequences found under {bronze_root}")
    LOG.info("WiPose: %d sequences; workers=%d", len(sequences), workers)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_wipose_sequence, item) for item in sequences]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res is not None:
                results.append(res)
            if i % 20 == 0:
                LOG.info("WiPose processed %d/%d sequences", i, len(sequences))
    if not results:
        raise RuntimeError("WiPose produced no windows")

    ds_dir = out_root / "wipose"
    ds_dir.mkdir(parents=True, exist_ok=True)
    x = np.concatenate([r["x"] for r in results], axis=0).astype(np.float32)
    pose = np.concatenate([r["pose"] for r in results], axis=0).astype(np.float32)
    labels = np.concatenate([r["action_label"] for r in results], axis=0).astype(np.int64)
    metadata = [m for r in results for m in r["metadata"]]
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(ds_dir / "x.npy", x)
    write_common_files(
        ds_dir,
        dataset="wipose",
        skeleton="wipose_eda18_pseudo3d",
        joint_names=WIPOSE_JOINTS,
        bones=WIPOSE_EDA_BONES,
        metadata=metadata,
        pose=pose,
        action_label=labels,
        action_mask=np.ones(len(labels), dtype=np.int64),
        source=str(bronze_root),
    )


def build_mmfi_from_gold(source_gold: Path, out_root: Path, chunk: int = 256) -> None:
    src = source_gold / "mmfi" if (source_gold / "mmfi").exists() else source_gold
    if not (src / "x.npy").exists() or not (src / "y.npz").exists():
        raise FileNotFoundError(f"MM-Fi source gold not found: {src}")
    ds_dir = out_root / "mmfi"
    ds_dir.mkdir(parents=True, exist_ok=True)
    x_src = np.load(src / "x.npy", mmap_mode="r")
    x_dst = np.lib.format.open_memmap(ds_dir / "x.npy", mode="w+", dtype=np.float32, shape=x_src.shape)
    for start in range(0, x_src.shape[0], chunk):
        block = np.array(x_src[start:start + chunk], dtype=np.float32, copy=True)
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        x_dst[start:start + chunk] = block
    del x_dst

    y = np.load(src / "y.npz")
    pose = np.asarray(y["pose"], dtype=np.float32)
    n = pose.shape[0]
    src_meta = np.load(src / "metadata.npz", allow_pickle=True)["metadata"]
    metadata = []
    labels = np.zeros(n, dtype=np.int64)
    group_to_action: dict[str, str] = {}
    old_rows: list[dict] = []
    for i in range(n):
        old = dict(src_meta[i]) if i < len(src_meta) else {}
        sid = old.get("sample_id", f"sample_{i:06d}")
        action = old.get("action") or (sid.split("_")[-1] if "_" in sid else "other")
        old_rows.append(old)
        group_to_action.setdefault(sid, action)

    for i, old in enumerate(old_rows):
        sid = old.get("sample_id", f"sample_{i:06d}")
        action = old.get("action") or (sid.split("_")[-1] if "_" in sid else "other")
        labels[i] = int(y["action_label"][i]) if "action_label" in y else ACTION_LABELS.get(action_from_sample(action), 0)
        subj = _MMFI_SID_RE.match(sid)
        old.update({
            "dataset": "mmfi",
            "sample_id": sid,
            "split": mmfi_subject_split(sid),
            "action": action,
            "split_group": f"S{subj.group(1)}" if subj else sid,
            "split_policy": "cross_subject_balanced_by_environment_v1",
        })
        metadata.append(old)
    y.close()
    write_common_files(
        ds_dir,
        dataset="mmfi",
        skeleton="h36m17",
        joint_names=H36M17_JOINTS,
        bones=H36M17_BONES,
        metadata=metadata,
        pose=pose,
        action_label=labels,
        action_mask=np.ones(n, dtype=np.int64),
        source=str(src),
    )


def write_root_manifest(out_root: Path) -> None:
    datasets = []
    for ds in sorted(out_root.iterdir()):
        if (ds / "manifest.json").exists():
            datasets.append(json.loads((ds / "manifest.json").read_text()))
    (out_root / "manifest.json").write_text(json.dumps({"datasets": datasets}, indent=2))
    (out_root / "label_maps.json").write_text(json.dumps({"activity": ACTION_LABELS}, indent=2))


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["wipose", "mmfi", "all"], default="all")
    ap.add_argument("--wipose-bronze", type=Path, default=Path("/opt/rfpose/data/bronze/public/wipose/raw_mat/Wi-Pose"))
    ap.add_argument("--mmfi-source-gold", type=Path, default=Path("data/gold/rfpose-mmfi-17j-v1"))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os_cpu_count() or 4))))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("wipose", "all"):
        build_wipose(args.wipose_bronze, args.out_root, args.workers)
    if args.dataset in ("mmfi", "all"):
        build_mmfi_from_gold(args.mmfi_source_gold, args.out_root)
    write_root_manifest(args.out_root)
    return 0


def os_cpu_count() -> int | None:
    try:
        import os
        return os.cpu_count()
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
