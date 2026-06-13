#!/usr/bin/env python3
"""Visualize WiPose bronze skeletons against WiPose model predictions.

This script is intended to run on the Eagle workspace where the WiPose gold
windows, bronze raw .mat files, and trained checkpoint are present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

try:
    import h5py
except ImportError:  # Some Eagle envs only have scipy.
    h5py = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfpose.models.wipose_net import WiPoseNet


WIPOSE_PARENT = {
    0: 1,
    2: 1, 3: 2, 4: 3, 5: 4,
    6: 1, 7: 6, 8: 7, 9: 8,
    10: 1, 11: 10, 12: 11, 13: 12,
    14: 1, 15: 14, 16: 15, 17: 16,
}
WIPOSE_BONES = [(parent, child) for child, parent in WIPOSE_PARENT.items()]

BODY_COLORS = {
    "head": "#1f77b4",
    "torso": "#2ca02c",
    "right_arm": "#ff7f0e",
    "left_arm": "#9467bd",
    "right_leg": "#d62728",
    "left_leg": "#17becf",
}


def bone_color(parent: int, child: int) -> str:
    pair = {parent, child}
    if pair <= {0, 1}:
        return BODY_COLORS["head"]
    if parent == 1 and child in {2, 6, 10, 14}:
        return BODY_COLORS["torso"]
    if child in {2, 3, 4, 5}:
        return BODY_COLORS["right_arm"]
    if child in {6, 7, 8, 9}:
        return BODY_COLORS["left_arm"]
    if child in {10, 11, 12, 13}:
        return BODY_COLORS["right_leg"]
    if child in {14, 15, 16, 17}:
        return BODY_COLORS["left_leg"]
    return "#555555"


def load_checkpoint(path: Path, device: torch.device) -> tuple[WiPoseNet, torch.Tensor, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = WiPoseNet(
        n_joints=data_cfg.get("n_joints", 18),
        n_antennas=model_cfg.get("n_antennas", 9),
        n_sub=model_cfg.get("n_sub", 30),
        n_packets=model_cfg.get("n_packets", 5),
        lstm_hidden=model_cfg.get("lstm_hidden", 544),
        lstm_layers=model_cfg.get("lstm_layers", 3),
        lstm_dropout=model_cfg.get("lstm_dropout", 0.1),
        cnn_dropout=model_cfg.get("cnn_dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, ckpt["ref_offsets"].to(device), cfg


def gold_csi_to_model_input(x_win: np.ndarray) -> np.ndarray:
    """Gold x window (2, T, 1350) -> WiPose model input (T, 9, 30, 5)."""
    amp_flat = np.nan_to_num(x_win[0].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return amp_flat.reshape(amp_flat.shape[0], 9, 30, 5)


def raw_skeleton_path(bronze_root: Path, sample_id: str, frame_num: int, split: str) -> Path:
    preferred = "Test" if split == "test" else "Train"
    candidates = [
        bronze_root / preferred / f"{sample_id}-frame{frame_num:03d}.mat",
        bronze_root / "Train" / f"{sample_id}-frame{frame_num:03d}.mat",
        bronze_root / "Test" / f"{sample_id}-frame{frame_num:03d}.mat",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def load_bronze_skeleton(path: Path) -> np.ndarray:
    """Load WiPose raw SkeletonPoints as (18, 3).

    WiPose .mat stores 54 values as x[18], y[18], z_or_conf[18].  The visualizer
    uses this layout because it produces a coherent human skeleton.
    """
    raw = None
    try:
        mat = loadmat(path)
        if "SkeletonPoints" in mat:
            raw = np.array(mat["SkeletonPoints"], dtype=np.float64).squeeze().reshape(-1)
    except NotImplementedError:
        raw = None

    if raw is None:
        if h5py is None:
            raise RuntimeError(f"{path} requires h5py, but h5py is not installed")
        with h5py.File(path, "r") as f:
            raw = np.array(f["SkeletonPoints"], dtype=np.float64).squeeze().reshape(-1)
    if raw.size != 54:
        raise ValueError(f"Expected 54 SkeletonPoints values in {path}, got {raw.size}")
    return raw.reshape(3, 18).T.astype(np.float32)


def bronze_to_display_pose(pose: np.ndarray) -> np.ndarray:
    """Convert raw WiPose x/y/depth-or-confidence into an upright 3D display pose."""
    x = pose[:, 0]
    y = pose[:, 1]
    z = pose[:, 2]

    body_h = max(float(y.max() - y.min()), 1.0)
    depth = (z - float(np.median(z))) * body_h * 0.35
    display = np.stack([x, depth, -y], axis=-1)
    display -= display[1:2]  # neck-centered
    return display


def orient_model_pose(pred: np.ndarray) -> np.ndarray:
    """Map model coordinates to a stable display orientation and center at neck."""
    out = pred.astype(np.float32).copy()
    out -= out[1:2]
    # Put the largest vertical-like spread on Z if needed. The trained target uses
    # WiPose's historical coordinate convention, so this keeps figures upright.
    spans = out.max(axis=0) - out.min(axis=0)
    vertical_axis = int(np.argmax(spans))
    if vertical_axis != 2:
        out[:, [2, vertical_axis]] = out[:, [vertical_axis, 2]]
    if out[0, 2] < out[1, 2]:
        out[:, 2] *= -1
    # Rotate in the display X/Z plane so Neck->Head points upward.  This makes
    # the WiPose convention readable as a human stick figure instead of a
    # diagonal body in the camera frame.
    up = out[0] - out[1]
    if np.linalg.norm(up[[0, 2]]) > 1e-6:
        angle = np.arctan2(up[0], up[2])
        c, s = np.cos(-angle), np.sin(-angle)
        x = out[:, 0].copy()
        z = out[:, 2].copy()
        out[:, 0] = c * x + s * z
        out[:, 2] = -s * x + c * z
    return out


def similarity_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Align src to dst with similarity transform for shape comparison."""
    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)
    x = src - src_mean
    y = dst - dst_mean
    src_norm = np.sqrt((x * x).sum())
    dst_norm = np.sqrt((y * y).sum())
    if src_norm < 1e-8 or dst_norm < 1e-8:
        return src - src_mean + dst_mean
    x /= src_norm
    y /= dst_norm
    u, _, vt = np.linalg.svd(x.T @ y)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = u @ vt
    scale = dst_norm / src_norm
    return (src - src_mean) @ r * scale + dst_mean


def set_equal_axes(ax, poses: list[np.ndarray]) -> None:
    pts = np.concatenate(poses, axis=0)
    center = pts.mean(axis=0)
    radius = max(float((pts.max(axis=0) - pts.min(axis=0)).max()) * 0.58, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=14, azim=-78)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("X")
    ax.set_ylabel("depth")
    ax.set_zlabel("up")


def draw_skeleton(ax, pose: np.ndarray, color: str, label: str, alpha: float, lw: float) -> None:
    for parent, child in WIPOSE_BONES:
        ax.plot(
            [pose[parent, 0], pose[child, 0]],
            [pose[parent, 1], pose[child, 1]],
            [pose[parent, 2], pose[child, 2]],
            color=color if color else bone_color(parent, child),
            linewidth=lw,
            alpha=alpha,
        )
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=22, color=color or "#222", alpha=alpha, label=label)
    ax.scatter([pose[0, 0]], [pose[0, 1]], [pose[0, 2]], s=70, color=color or "#222", alpha=alpha)


def plot_sample(out_path: Path, title: str, bronze: np.ndarray, pred: np.ndarray, gold: np.ndarray) -> float:
    pred_aligned = similarity_align(pred, bronze)
    gold_disp = similarity_align(orient_model_pose(gold), bronze)
    shape_err = float(np.linalg.norm(pred_aligned - bronze, axis=-1).mean())

    fig = plt.figure(figsize=(14, 6), facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    draw_skeleton(ax1, bronze, "#1f77b4", "Bronze raw GT", 0.95, 3.2)
    set_equal_axes(ax1, [bronze])
    ax1.set_title("Bronze-derived WiPose GT skeleton", fontsize=12, fontweight="bold")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_skeleton(ax2, bronze, "#1f77b4", "Bronze raw GT", 0.55, 3.4)
    draw_skeleton(ax2, pred_aligned, "#d62728", "Model prediction", 0.92, 2.6)
    set_equal_axes(ax2, [bronze, pred_aligned, gold_disp])
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_title(f"Model over Bronze GT | aligned err={shape_err:.1f}px", fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return shape_err


def select_indices(metadata: np.ndarray, requested: list[str], max_samples: int) -> list[int]:
    indices: list[int] = []
    used_actions: set[str] = set()
    for i, meta in enumerate(metadata):
        sample_id = meta["sample_id"]
        action = sample_id.split("_")[0]
        if meta.get("split") != "test":
            continue
        if requested and sample_id not in requested and action not in requested:
            continue
        if not requested and action in used_actions:
            continue
        indices.append(i)
        used_actions.add(action)
        if len(indices) >= max_samples:
            break
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/data/gold/rfpose-wipose-18j-v1/wipose")
    parser.add_argument("--bronze-root", default="/opt/rfpose/data/bronze/public/wipose/raw_mat/Wi-Pose")
    parser.add_argument("--checkpoint", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/ml/checkpoints/wipose_best.pt")
    parser.add_argument("--output-dir", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/viz_output/wipose_bronze_model")
    parser.add_argument("--samples", nargs="*", default=[])
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--frame-offset", type=int, default=30)
    parser.add_argument(
        "--display-source",
        choices=["raw", "gold"],
        default="raw",
        help="raw draws SkeletonPoints from Bronze .mat; gold draws the Bronze-derived normalized GT used for training.",
    )
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    bronze_root = Path(args.bronze_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ref_offsets, _ = load_checkpoint(Path(args.checkpoint), device)

    x = np.load(gold_dir / "x.npy", mmap_mode="r")
    y = np.load(gold_dir / "y.npz")
    poses = y["pose"]
    metadata = np.load(gold_dir / "metadata.npz", allow_pickle=True)["metadata"]
    indices = select_indices(metadata, args.samples, args.max_samples)
    if not indices:
        raise SystemExit("No matching test samples found")

    summary: list[tuple[str, Path, float]] = []
    for idx in indices:
        meta = metadata[idx]
        csi = gold_csi_to_model_input(np.asarray(x[idx]))
        with torch.no_grad():
            inp = torch.from_numpy(csi).unsqueeze(0).to(device)
            pred_seq = model(inp, ref_offsets)["coords"].squeeze(0).cpu().numpy()

        frame_offset = min(args.frame_offset, pred_seq.shape[0] - 1)
        frame_num = int(meta["window_start"]) + frame_offset + 1
        if args.display_source == "gold":
            bronze = orient_model_pose(poses[idx, frame_offset])
        else:
            mat_path = raw_skeleton_path(bronze_root, meta["sample_id"], frame_num, meta.get("split", "test"))
            bronze = bronze_to_display_pose(load_bronze_skeleton(mat_path))
        pred = orient_model_pose(pred_seq[frame_offset])
        gold = poses[idx, frame_offset]

        out_path = out_dir / f"pred_vs_bronze_{idx:04d}_{meta['sample_id']}_f{frame_num:03d}.png"
        err = plot_sample(
            out_path,
            f"{meta['sample_id']} | window={meta['window_start']} | frame={frame_num}",
            bronze,
            pred,
            gold,
        )
        summary.append((f"{meta['sample_id']} frame {frame_num}", out_path, err))
        print(f"saved {out_path} err_px={err:.2f}")

    # Overview sheet.
    n = len(summary)
    fig = plt.figure(figsize=(5 * n, 5), facecolor="white")
    for col, (name, _, _) in enumerate(summary, 1):
        idx = indices[col - 1]
        meta = metadata[idx]
        frame_offset = min(args.frame_offset, poses.shape[1] - 1)
        frame_num = int(meta["window_start"]) + frame_offset + 1
        if args.display_source == "gold":
            bronze = orient_model_pose(poses[idx, frame_offset])
        else:
            mat_path = raw_skeleton_path(bronze_root, meta["sample_id"], frame_num, meta.get("split", "test"))
            bronze = bronze_to_display_pose(load_bronze_skeleton(mat_path))
        csi = gold_csi_to_model_input(np.asarray(x[idx]))
        with torch.no_grad():
            pred_seq = model(torch.from_numpy(csi).unsqueeze(0).to(device), ref_offsets)["coords"].squeeze(0).cpu().numpy()
        pred = similarity_align(orient_model_pose(pred_seq[frame_offset]), bronze)
        ax = fig.add_subplot(1, n, col, projection="3d")
        draw_skeleton(ax, bronze, "#1f77b4", "Bronze", 0.50, 3.0)
        draw_skeleton(ax, pred, "#d62728", "Pred", 0.90, 2.4)
        set_equal_axes(ax, [bronze, pred])
        ax.set_title(name, fontsize=10, fontweight="bold")
    fig.suptitle("WiPose Bronze raw GT (blue) vs model prediction (red)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    overview = out_dir / "overview_pred_vs_bronze.png"
    fig.savefig(overview, dpi=180)
    plt.close(fig)
    print(f"saved {overview}")


if __name__ == "__main__":
    main()
