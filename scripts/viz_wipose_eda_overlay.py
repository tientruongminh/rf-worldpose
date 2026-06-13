#!/usr/bin/env python3
"""WiPose prediction-vs-Bronze visualization in the original EDA style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfpose.models.wipose_net import WiPoseNet


# Exact bone layout used by ml/eda/eda_wipose.py.
WIPOSE_EDA_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (8, 11), (0, 14), (14, 16), (0, 15), (15, 17),
]


def get_bone_color(u: int, v: int) -> str:
    right_side = {2, 3, 4, 5, 6, 7, 14, 16}
    left_side = {8, 9, 10, 11, 12, 13, 15, 17}
    if u in right_side or v in right_side:
        return "#ff7f0e"
    if u in left_side or v in left_side:
        return "#1f77b4"
    return "#2ca02c"


def load_model(checkpoint: Path, device: torch.device) -> tuple[WiPoseNet, torch.Tensor, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    d = cfg["data"]
    model = WiPoseNet(
        n_joints=d.get("n_joints", 18),
        n_antennas=m.get("n_antennas", 9),
        n_sub=m.get("n_sub", 30),
        n_packets=m.get("n_packets", 5),
        lstm_hidden=m.get("lstm_hidden", 544),
        lstm_layers=m.get("lstm_layers", 3),
        lstm_dropout=m.get("lstm_dropout", 0.1),
        cnn_dropout=m.get("cnn_dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["ref_offsets"].to(device), cfg


def gold_csi_to_model_input(x_win: np.ndarray) -> np.ndarray:
    amp_flat = np.nan_to_num(x_win[0].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return amp_flat.reshape(amp_flat.shape[0], 9, 30, 5)


def raw_mat_path(bronze_root: Path, sample_id: str, frame_num: int, split: str) -> Path:
    preferred = "Test" if split == "test" else "Train"
    candidates = [
        bronze_root / preferred / f"{sample_id}-frame{frame_num:03d}.mat",
        bronze_root / "Train" / f"{sample_id}-frame{frame_num:03d}.mat",
        bronze_root / "Test" / f"{sample_id}-frame{frame_num:03d}.mat",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(candidates[0])


def load_raw_skeleton_eda(path: Path) -> np.ndarray:
    """Return EDA raw skeleton shape (3,18): rows are x_pixel, y_pixel, confidence/depth."""
    with h5py.File(path, "r") as f:
        raw = np.array(f["SkeletonPoints"], dtype=np.float64).squeeze()
    if raw.shape == (18, 3):
        raw = raw.T
    elif raw.size == 54:
        raw = raw.reshape(3, 18)
    else:
        raise ValueError(f"Unexpected SkeletonPoints shape {raw.shape} in {path}")
    return raw.astype(np.float32)


def eda_display_from_raw(skel_3x18: np.ndarray) -> np.ndarray:
    """Exact EDA display mapping: X=y_pixel, Y=0, Z=-x_pixel."""
    x_pixels = skel_3x18[0]
    y_pixels = skel_3x18[1]
    x3 = y_pixels
    y3 = np.zeros_like(x_pixels)
    z3 = -x_pixels
    pts = np.stack([x3, y3, z3], axis=-1).astype(np.float32)
    return pts


def orient_pred_to_display(pred: np.ndarray) -> np.ndarray:
    """Make model output comparable before similarity-aligning to raw EDA plane."""
    p = pred.astype(np.float32).copy()
    p -= p.mean(axis=0, keepdims=True)
    spans = p.max(axis=0) - p.min(axis=0)
    vertical = int(np.argmax(spans))
    if vertical != 2:
        p[:, [2, vertical]] = p[:, [vertical, 2]]
    return p


def similarity_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_m = src.mean(axis=0, keepdims=True)
    dst_m = dst.mean(axis=0, keepdims=True)
    x = src - src_m
    y = dst - dst_m
    xn = np.sqrt((x * x).sum())
    yn = np.sqrt((y * y).sum())
    if xn < 1e-8 or yn < 1e-8:
        return src - src_m + dst_m
    x0, y0 = x / xn, y / yn
    u, _, vt = np.linalg.svd(x0.T @ y0)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = u @ vt
    return (src - src_m) @ r * (yn / xn) + dst_m


def draw_gt(ax, raw_skel: np.ndarray) -> None:
    pts = eda_display_from_raw(raw_skel)
    conf = raw_skel[2]
    node_colors = plt.cm.tab20(np.linspace(0, 1, 18))
    sizes = np.clip(conf, 0.2, 1.0) * 70
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=node_colors, s=sizes, zorder=5, edgecolors="k", linewidths=0.5)
    for u, v in WIPOSE_EDA_BONES:
        if conf[u] > 0.1 and conf[v] > 0.1:
            ax.plot(
                [pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]], [pts[u, 2], pts[v, 2]],
                color=get_bone_color(u, v), linewidth=2.7, alpha=0.82, zorder=3,
            )


def draw_pred(ax, pred_aligned: np.ndarray) -> None:
    ax.scatter(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2], c="#d62728", s=34, zorder=7, edgecolors="white", linewidths=0.4)
    for u, v in WIPOSE_EDA_BONES:
        ax.plot(
            [pred_aligned[u, 0], pred_aligned[v, 0]],
            [pred_aligned[u, 1], pred_aligned[v, 1]],
            [pred_aligned[u, 2], pred_aligned[v, 2]],
            color="#d62728", linewidth=2.0, alpha=0.90, zorder=6,
        )


def set_eda_axes(ax, raw_seq_pts: list[np.ndarray], pred_pts: list[np.ndarray] | None = None) -> None:
    pts = raw_seq_pts[:]
    if pred_pts:
        pts += pred_pts
    all_pts = np.concatenate(pts, axis=0)
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
    max_range = max(float((all_pts.max(axis=0) - all_pts.min(axis=0)).max()) / 2.0, 1.0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X (Width)", fontsize=8)
    ax.set_ylabel("Y (Depth)", fontsize=8)
    ax.set_zlabel("Z (Height)", fontsize=8)
    ax.view_init(elev=10, azim=-60)
    ax.grid(True, linestyle="--", alpha=0.2)


def plot_window(out_path: Path, meta: dict, raw_skeletons: list[np.ndarray], pred_seq: np.ndarray, frame_offsets: np.ndarray) -> list[dict]:
    raw_display = [eda_display_from_raw(s) for s in raw_skeletons]
    pred_display = []
    records = []
    for skel, f in zip(raw_skeletons, frame_offsets):
        gt_pts = eda_display_from_raw(skel)
        pred_pts = similarity_align(orient_pred_to_display(pred_seq[int(f)]), gt_pts)
        pred_display.append(pred_pts)
        err = float(np.linalg.norm(pred_pts - gt_pts, axis=-1).mean())
        records.append({"frame_offset": int(f), "aligned_pixel_error": round(err, 1)})

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")
    for i, (skel, pred_pts, rec) in enumerate(zip(raw_skeletons, pred_display, records), 1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        ax.set_facecolor("white")
        draw_gt(ax, skel)
        draw_pred(ax, pred_pts)
        set_eda_axes(ax, raw_display, pred_display)
        progress = int((rec["frame_offset"] / max(pred_seq.shape[0] - 1, 1)) * 100)
        ax.set_title(
            f"Frame {rec['frame_offset']} ({progress}%) | err={rec['aligned_pixel_error']:.1f}px",
            fontsize=11, fontweight="bold", color="#d62728",
        )
    fig.suptitle(
        f"WiPose Predict vs Bronze GT - EDA style\n{meta['sample_id']} | GT=color skeleton, Pred=red",
        fontsize=16, fontweight="bold", y=0.98,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", default="/root/rf-worldpose/data/gold/rfpose-wipose-18j-v1/wipose")
    parser.add_argument("--bronze-root", default="/opt/rfpose/data/bronze/public/wipose/raw_mat/Wi-Pose")
    parser.add_argument("--checkpoint", default="/root/rf-worldpose/ml/checkpoints/wipose_best.pt")
    parser.add_argument("--output-dir", default="/root/rf-worldpose/viz_output/wipose_eda_overlay")
    parser.add_argument("--samples", nargs="*", default=[])
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_dir = Path(args.gold_dir)
    bronze_root = Path(args.bronze_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ref_offsets, _ = load_model(Path(args.checkpoint), device)

    x = np.load(gold_dir / "x.npy", mmap_mode="r")
    metadata = np.load(gold_dir / "metadata.npz", allow_pickle=True)["metadata"]

    indices = []
    if args.samples:
        for token in args.samples:
            for i, m in enumerate(metadata):
                sample_id = m["sample_id"]
                action = sample_id.split("_")[0]
                if sample_id == token or action == token:
                    indices.append(i)
                    break
            if len(indices) >= args.num_samples:
                break
    else:
        seen = set()
        for i, m in enumerate(metadata):
            if m.get("split") != "test":
                continue
            action = m["sample_id"].split("_")[0]
            if action in seen:
                continue
            indices.append(i)
            seen.add(action)
            if len(indices) >= args.num_samples:
                break

    results = []
    for idx in indices:
        meta = metadata[idx]
        csi = gold_csi_to_model_input(np.asarray(x[idx]))
        with torch.no_grad():
            pred_seq = model(torch.from_numpy(csi).unsqueeze(0).to(device), ref_offsets)["coords"].squeeze(0).cpu().numpy()

        frame_offsets = np.linspace(0, pred_seq.shape[0] - 1, 6, dtype=int)
        raw_skeletons = []
        for f in frame_offsets:
            frame_num = int(meta["window_start"]) + int(f) + 1
            raw_skeletons.append(load_raw_skeleton_eda(raw_mat_path(bronze_root, meta["sample_id"], frame_num, meta.get("split", "test"))))
        out_path = out_dir / f"wipose_eda_overlay_{idx:04d}_{meta['sample_id']}.png"
        records = plot_window(out_path, meta, raw_skeletons, pred_seq, frame_offsets)
        print(f"saved {out_path}")
        results.append({"idx": int(idx), "metadata": dict(meta), "file": out_path.name, "frames": records})
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
