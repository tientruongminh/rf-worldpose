"""ETL adapters: Bronze → Gold NPZ for public WiFi-CSI datasets.

Each adapter reads a dataset's raw files and produces the unified Gold format:
    x.npy          [N, 2, T, N_sub_padded]   (amplitude, phase channels)
    y.npz          pose [N, J, 3], pose_mask [N], activity [N], activity_id [N], ...
    metadata.npz   per-window metadata (dataset, sample_id, split, window_start)
    label_maps.json   activity name → id mapping

Target skeleton (13 joints):
    head, left_shoulder, right_shoulder, left_elbow, right_elbow,
    left_wrist, right_wrist, left_hip, right_hip, left_knee,
    right_knee, left_ankle, right_ankle
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

GOLD_JOINTS = [
    "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
N_JOINTS = len(GOLD_JOINTS)
WINDOW_FRAMES = 60
N_SUB_PADDED = 270
STRIDE = 21
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def _pad_or_truncate_subs(csi: np.ndarray, target: int = N_SUB_PADDED) -> np.ndarray:
    """Pad or truncate subcarrier dimension to target size."""
    n = csi.shape[-1]
    if n >= target:
        return csi[..., :target]
    pad_width = [(0, 0)] * (csi.ndim - 1) + [(0, target - n)]
    return np.pad(csi, pad_width, mode="constant")


def _sliding_windows(seq: np.ndarray, window: int, stride: int) -> list[np.ndarray]:
    """Create sliding windows along axis 0."""
    out = []
    for start in range(0, len(seq) - window + 1, stride):
        out.append(seq[start : start + window])
    return out


def _assign_splits(n: int, seed: int = 42) -> list[str]:
    """Assign train/val/test splits."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * TEST_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    splits = ["train"] * n
    for i in idx[:n_test]:
        splits[i] = "test"
    for i in idx[n_test : n_test + n_val]:
        splits[i] = "val"
    return splits


def _remap_joints(pose_3d: np.ndarray, src_names: list[str]) -> np.ndarray:
    """Remap arbitrary skeleton to 13-joint Gold format. Missing joints = 0."""
    src_map = {name.lower().replace(" ", "_"): i for i, name in enumerate(src_names)}
    aliases = {
        "head": ["head", "nose", "head_top", "skull"],
        "left_shoulder": ["left_shoulder", "l_shoulder", "lshoulder"],
        "right_shoulder": ["right_shoulder", "r_shoulder", "rshoulder"],
        "left_elbow": ["left_elbow", "l_elbow", "lelbow"],
        "right_elbow": ["right_elbow", "r_elbow", "relbow"],
        "left_wrist": ["left_wrist", "l_wrist", "lwrist", "left_hand"],
        "right_wrist": ["right_wrist", "r_wrist", "rwrist", "right_hand"],
        "left_hip": ["left_hip", "l_hip", "lhip"],
        "right_hip": ["right_hip", "r_hip", "rhip"],
        "left_knee": ["left_knee", "l_knee", "lknee"],
        "right_knee": ["right_knee", "r_knee", "rknee"],
        "left_ankle": ["left_ankle", "l_ankle", "lankle", "left_foot"],
        "right_ankle": ["right_ankle", "r_ankle", "rankle", "right_foot"],
    }
    T = pose_3d.shape[0] if pose_3d.ndim == 3 else 1
    out = np.zeros((T, N_JOINTS, 3), dtype=np.float32)
    p = pose_3d.reshape(T, -1, 3)
    for j, joint in enumerate(GOLD_JOINTS):
        for alias in aliases[joint]:
            if alias in src_map:
                out[:, j, :] = p[:, src_map[alias], :]
                break
    return out


def _save_gold(
    out_dir: Path,
    x_list: list[np.ndarray],
    pose_list: list[np.ndarray],
    activity_ids: list[int],
    metadata: list[dict],
    label_map: dict[str, int],
) -> dict:
    """Save arrays in Gold NPZ format."""
    out_dir.mkdir(parents=True, exist_ok=True)

    x_all = np.stack(x_list).astype(np.float32)
    pose_all = np.stack(pose_list).astype(np.float32)
    n = len(x_all)

    splits = _assign_splits(n)
    for i, m in enumerate(metadata):
        m["split"] = splits[i]

    activity_arr = np.array(activity_ids, dtype=np.int64)
    pose_mask = np.ones(n, dtype=np.int64)

    np.save(out_dir / "x.npy", x_all)
    np.savez_compressed(
        out_dir / "y.npz",
        pose=pose_all,
        pose_mask=pose_mask,
        activity=np.array([label_map.get(str(a), 0) for a in activity_ids], dtype=np.int64),
        activity_id=activity_arr,
        activity_mask=np.ones(n, dtype=np.int64),
        location_id=np.zeros(n, dtype=np.int64),
        location_mask=np.zeros(n, dtype=np.int64),
        environment_id=np.zeros(n, dtype=np.int64),
        environment_mask=np.zeros(n, dtype=np.int64),
        subject_id=np.zeros(n, dtype=np.int64),
        subject_mask=np.zeros(n, dtype=np.int64),
    )
    np.savez_compressed(
        out_dir / "metadata.npz",
        metadata=np.array(metadata, dtype=object),
    )

    mean_val = float(x_all.mean())
    std_val = float(x_all.std() + 1e-6)
    (out_dir / "normalization.json").write_text(json.dumps({"mean": mean_val, "std": std_val}, indent=2))

    stats = {
        "dataset": out_dir.name,
        "num_samples": n,
        "splits": {s: splits.count(s) for s in ("train", "val", "test")},
        "x_shape": list(x_all.shape),
        "pose_shape": list(pose_all.shape),
        "window_frames": WINDOW_FRAMES,
        "stride": STRIDE,
        "pose_joints": GOLD_JOINTS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "label_maps.json").write_text(json.dumps({"activity": label_map}, indent=2))

    log.info("%s: %d windows, x=%s", out_dir.name, n, x_all.shape)
    return stats


# ─────────────────────────────────────────────────────────────────────
# MMFi adapter
# ─────────────────────────────────────────────────────────────────────
MMFI_JOINTS_17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

MMFI_ACTIONS = {
    "A01": "walk", "A02": "run", "A03": "jump", "A04": "squat",
    "A05": "sit_down", "A06": "stand_up", "A07": "fall",
    "A08": "bend", "A09": "hand_clap", "A10": "wave",
    "A11": "phone_call", "A12": "drink_water", "A13": "throw",
    "A14": "pick_up", "A15": "push", "A16": "pull",
    "A17": "kick", "A18": "toss_paper", "A19": "draw_x",
    "A20": "draw_tick", "A21": "rotation", "A22": "nothing",
    "A23": "nothing", "A24": "nothing", "A25": "nothing",
    "A26": "nothing", "A27": "nothing",
}


def adapt_mmfi(bronze_dir: str | Path, gold_dir: str | Path) -> dict:
    """Convert MMFi bronze to Gold NPZ format."""
    import scipy.io as sio

    bronze = Path(bronze_dir) / "MMFi_Dataset"
    if not bronze.exists():
        bronze = Path(bronze_dir)
    out = Path(gold_dir)

    x_list, pose_list, act_ids, meta = [], [], [], []
    label_map = {}

    for env_dir in sorted(bronze.iterdir()):
        if not env_dir.is_dir():
            continue
        for subj_dir in sorted(env_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            for action_dir in sorted(subj_dir.iterdir()):
                if not action_dir.is_dir():
                    continue
                csi_dir = action_dir / "wifi-csi"
                gt_path = action_dir / "ground_truth.npy"
                if not csi_dir.exists() or not gt_path.exists():
                    continue

                gt = np.load(gt_path, allow_pickle=True)  # (N_frames, 17, 3)
                n_frames = gt.shape[0]
                csi_files = sorted(csi_dir.glob("frame*.mat"))
                if len(csi_files) < WINDOW_FRAMES:
                    continue

                amp_seq, phase_seq = [], []
                for cf in csi_files[:n_frames]:
                    mat = sio.loadmat(str(cf))
                    amp = mat["CSIamp"].astype(np.float32)    # (3, 114, 10)
                    phase = mat["CSIphase"].astype(np.float32)
                    amp_flat = amp.reshape(-1)     # 3*114*10 = 3420 → pad to 270
                    phase_flat = phase.reshape(-1)
                    amp_seq.append(amp_flat)
                    phase_seq.append(phase_flat)

                amp_arr = np.stack(amp_seq)    # (T, 3420)
                phase_arr = np.stack(phase_seq)

                action_name = MMFI_ACTIONS.get(action_dir.name, "nothing")
                if action_name not in label_map:
                    label_map[action_name] = len(label_map)
                act_id = label_map[action_name]

                sample_id = f"{env_dir.name}_{subj_dir.name}_{action_dir.name}"
                pose_remapped = _remap_joints(gt, MMFI_JOINTS_17)  # (T, 13, 3)

                for start in range(0, len(amp_arr) - WINDOW_FRAMES + 1, STRIDE):
                    end = start + WINDOW_FRAMES
                    a_win = _pad_or_truncate_subs(amp_arr[start:end], N_SUB_PADDED)
                    p_win = _pad_or_truncate_subs(phase_arr[start:end], N_SUB_PADDED)
                    csi_win = np.stack([a_win, p_win], axis=0)  # (2, T, 270)

                    x_list.append(csi_win)
                    pose_list.append(pose_remapped[start:end])  # (T, 13, 3)
                    act_ids.append(act_id)
                    meta.append({"dataset": "mmfi", "sample_id": sample_id, "window_start": start})

    if not x_list:
        log.warning("MMFi: no windows produced")
        return {"num_samples": 0}

    return _save_gold(out, x_list, pose_list, act_ids, meta, label_map)


# ─────────────────────────────────────────────────────────────────────
# Wi-Pose adapter
# ─────────────────────────────────────────────────────────────────────
WIPOSE_JOINTS_18 = [
    "head", "neck", "right_shoulder", "right_elbow", "right_wrist", "right_hand",
    "left_shoulder", "left_elbow", "left_wrist", "left_hand",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "left_hip", "left_knee", "left_ankle", "left_foot",
]


def adapt_wipose(bronze_dir: str | Path, gold_dir: str | Path) -> dict:
    """Convert Wi-Pose bronze (.mat HDF5) to Gold NPZ format."""
    import h5py

    bronze = Path(bronze_dir)
    out = Path(gold_dir)

    x_list, pose_list, act_ids, meta = [], [], [], []
    label_map = {}

    for split_name in ["Train", "Test"]:
        split_dir = bronze / split_name
        if not split_dir.exists():
            continue

        sequences: dict[str, list] = {}
        for mat_file in sorted(split_dir.glob("*.mat")):
            parts = mat_file.stem.rsplit("-", 1)
            if len(parts) != 2:
                continue
            seq_id, frame_str = parts
            frame_num = int(frame_str.replace("frame", ""))
            sequences.setdefault(seq_id, []).append((frame_num, mat_file))

        for seq_id, frames in sorted(sequences.items()):
            frames.sort(key=lambda x: x[0])
            if len(frames) < WINDOW_FRAMES:
                continue

            action_name = seq_id.rsplit("_", 1)[0]
            if action_name not in label_map:
                label_map[action_name] = len(label_map)
            act_id = label_map[action_name]

            csi_frames, pose_frames = [], []
            for _, mat_path in frames:
                try:
                    with h5py.File(str(mat_path), "r") as f:
                        csi = np.array(f["CSI"], dtype=np.float64)   # (3, 3, 30, 5)
                        skel = np.array(f["SkeletonPoints"], dtype=np.float64)  # (1, 54)
                except Exception:
                    continue

                amp = np.abs(csi).reshape(-1).astype(np.float32)    # 3*3*30*5 = 1350
                phase = np.angle(csi).reshape(-1).astype(np.float32) if np.iscomplexobj(csi) else np.zeros_like(amp)
                csi_frames.append(np.stack([amp, phase]))  # (2, 1350)

                joints_3d = skel.reshape(-1, 3).astype(np.float32)  # (18, 3)
                pose_frames.append(joints_3d)

            if len(csi_frames) < WINDOW_FRAMES:
                continue

            csi_arr = np.stack(csi_frames)    # (T, 2, 1350)
            pose_arr = np.stack(pose_frames)  # (T, 18, 3)
            pose_remapped = _remap_joints(pose_arr, WIPOSE_JOINTS_18)  # (T, 13, 3)

            for start in range(0, len(csi_arr) - WINDOW_FRAMES + 1, STRIDE):
                end = start + WINDOW_FRAMES
                win = csi_arr[start:end]  # (60, 2, 1350)
                amp_win = _pad_or_truncate_subs(win[:, 0, :], N_SUB_PADDED)   # (60, 270)
                phase_win = _pad_or_truncate_subs(win[:, 1, :], N_SUB_PADDED)
                x = np.stack([amp_win, phase_win], axis=0)  # (2, 60, 270)

                x_list.append(x)
                pose_list.append(pose_remapped[start:end])
                act_ids.append(act_id)
                meta.append({"dataset": "wipose", "sample_id": seq_id, "window_start": start})

    if not x_list:
        log.warning("Wi-Pose: no windows produced")
        return {"num_samples": 0}

    return _save_gold(out, x_list, pose_list, act_ids, meta, label_map)


# ─────────────────────────────────────────────────────────────────────
# wifipose_dataset (Person-in-WiFi) adapter
# ─────────────────────────────────────────────────────────────────────
WIFIPOSE_JOINTS_14 = [
    "head", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
]


def adapt_wifipose(bronze_dir: str | Path, gold_dir: str | Path) -> dict:
    """Convert wifipose_dataset (Person-in-WiFi) to Gold NPZ format."""
    import h5py

    bronze = Path(bronze_dir)
    out = Path(gold_dir)

    x_list, pose_list, act_ids, meta = [], [], [], []

    for split_name in ["train_data", "test_data", "val_data"]:
        split_dir = bronze / split_name
        if not split_dir.exists():
            continue

        csi_dir = split_dir / "csi"
        kp_dir = split_dir / "keypoint"
        if not csi_dir.exists() or not kp_dir.exists():
            continue

        sequences: dict[str, list] = {}
        for csi_file in sorted(csi_dir.glob("*.mat")):
            parts = csi_file.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            seq_id = parts[0]
            frame_id = int(parts[1])
            kp_file = kp_dir / (csi_file.stem + ".npy")
            if not kp_file.exists():
                continue
            sequences.setdefault(seq_id, []).append((frame_id, csi_file, kp_file))

        for seq_id, frames in sorted(sequences.items()):
            frames.sort(key=lambda x: x[0])
            if len(frames) < WINDOW_FRAMES:
                continue

            csi_frames, pose_frames = [], []
            for _, csi_path, kp_path in frames:
                try:
                    with h5py.File(str(csi_path), "r") as f:
                        csi_raw = np.array(f["csi_out"])  # (20, 30, 3, 3) complex struct
                    if csi_raw.dtype.names:
                        csi_complex = csi_raw["real"] + 1j * csi_raw["imag"]
                    else:
                        csi_complex = csi_raw.astype(np.complex128)

                    amp = np.abs(csi_complex).reshape(-1).astype(np.float32)   # 20*30*3*3 = 5400
                    phase = np.angle(csi_complex).reshape(-1).astype(np.float32)
                except Exception:
                    continue

                kp = np.load(str(kp_path)).astype(np.float32)  # (1, 14, 3)
                kp = kp.reshape(-1, 3)  # (14, 3)

                csi_frames.append(np.stack([amp, phase]))
                pose_frames.append(kp)

            if len(csi_frames) < WINDOW_FRAMES:
                continue

            csi_arr = np.stack(csi_frames)    # (T, 2, 5400)
            pose_arr = np.stack(pose_frames)  # (T, 14, 3)
            pose_remapped = _remap_joints(pose_arr, WIFIPOSE_JOINTS_14)

            for start in range(0, len(csi_arr) - WINDOW_FRAMES + 1, STRIDE):
                end = start + WINDOW_FRAMES
                win = csi_arr[start:end]
                amp_win = _pad_or_truncate_subs(win[:, 0, :], N_SUB_PADDED)
                phase_win = _pad_or_truncate_subs(win[:, 1, :], N_SUB_PADDED)
                x = np.stack([amp_win, phase_win], axis=0)

                x_list.append(x)
                pose_list.append(pose_remapped[start:end])
                act_ids.append(0)
                meta.append({"dataset": "wifipose", "sample_id": seq_id, "window_start": start})

    if not x_list:
        log.warning("wifipose: no windows produced")
        return {"num_samples": 0}

    return _save_gold(out, x_list, pose_list, act_ids, meta, {"pose_only": 0})


# ─────────────────────────────────────────────────────────────────────
# Unified builder
# ─────────────────────────────────────────────────────────────────────
def build_all_gold(bronze_root: str | Path, gold_root: str | Path) -> dict:
    """Run all adapters and write to gold_root/{dataset_name}/."""
    bronze = Path(bronze_root)
    gold = Path(gold_root)
    gold.mkdir(parents=True, exist_ok=True)

    results = {}
    adapters = [
        ("mmfi", "MMFi_Dataset", adapt_mmfi),
        ("wipose", "Wi-Pose/Wi-Pose", adapt_wipose),
        ("wifipose", "wifipose_dataset", adapt_wifipose),
    ]

    for name, subdir, fn in adapters:
        src = bronze / subdir
        if not src.exists():
            log.warning("Skipping %s: %s not found", name, src)
            continue
        log.info("Processing %s from %s ...", name, src)
        try:
            stats = fn(src, gold / name)
            results[name] = stats
            log.info("  → %s: %d samples", name, stats.get("num_samples", 0))
        except Exception:
            log.exception("Failed to process %s", name)
            results[name] = {"error": True}

    summary = {
        "datasets": results,
        "num_datasets": len([r for r in results.values() if not r.get("error")]),
        "num_samples": sum(r.get("num_samples", 0) for r in results.values()),
    }
    (gold / "summary_new.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Convert bronze datasets to Gold NPZ")
    ap.add_argument("--bronze-root", required=True, help="Path to data/bronze/")
    ap.add_argument("--gold-root", required=True, help="Output Gold directory")
    ap.add_argument("--dataset", choices=["mmfi", "wipose", "wifipose", "all"], default="all")
    args = ap.parse_args()

    if args.dataset == "all":
        summary = build_all_gold(args.bronze_root, args.gold_root)
    elif args.dataset == "mmfi":
        summary = adapt_mmfi(
            Path(args.bronze_root) / "MMFi_Dataset", Path(args.gold_root) / "mmfi"
        )
    elif args.dataset == "wipose":
        summary = adapt_wipose(
            Path(args.bronze_root) / "Wi-Pose/Wi-Pose", Path(args.gold_root) / "wipose"
        )
    elif args.dataset == "wifipose":
        summary = adapt_wifipose(
            Path(args.bronze_root) / "wifipose_dataset", Path(args.gold_root) / "wifipose"
        )

    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary.get("num_samples", 0) > 0 else 1)
