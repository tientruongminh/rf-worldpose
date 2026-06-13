#!/usr/bin/env python3
"""Visualize the best MM-Fi 17-joint checkpoint.

Runs on the Eagle workspace, where the checkpoint and Gold NPZ dataset live.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfpose.data.gold_npz_dataset import GoldNpzDataset
from rfpose.models.vit2d_pose import CSIViT2DPose


H36M_PARENT = {
    1: 0, 2: 1, 3: 2,
    4: 0, 5: 4, 6: 5,
    7: 0, 8: 7,
    9: 8, 10: 9,
    11: 8, 12: 11, 13: 12,
    14: 8, 15: 14, 16: 15,
}
H36M_BONES = [(p, c) for c, p in H36M_PARENT.items()]


def load_model(checkpoint: Path, device: torch.device) -> tuple[CSIViT2DPose, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    d = cfg["data"]
    m = cfg["model"]
    model = CSIViT2DPose(
        n_subcarriers=d.get("n_subcarriers", 342),
        patch_freq=m.get("patch_size", 6),
        d_model=m.get("d_model", 256),
        n_layers=m.get("n_spatial_layers", 4),
        n_heads=m.get("spatial_heads", 8),
        n_joints=d.get("n_joints", 17),
        num_actions=m.get("num_actions", 28),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def orient_pose(pose: np.ndarray) -> np.ndarray:
    """Center at pelvis and rotate so torso points upward in display space."""
    out = pose.astype(np.float32).copy()
    out -= out[0:1]

    # Pick the strongest torso-up vector available: pelvis->thorax/head.
    up = out[10] - out[0] if np.linalg.norm(out[10] - out[0]) > 1e-6 else out[8] - out[0]
    if up[2] < 0:
        out[:, 2] *= -1
        up = out[10] - out[0]

    # Rotate around Y so the body is less diagonally tilted in the X/Z view.
    if np.linalg.norm(up[[0, 2]]) > 1e-6:
        angle = np.arctan2(up[0], up[2])
        c, s = np.cos(-angle), np.sin(-angle)
        x = out[:, 0].copy()
        z = out[:, 2].copy()
        out[:, 0] = c * x + s * z
        out[:, 2] = -s * x + c * z
    return out


def similarity_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)
    x = src - src_mean
    y = dst - dst_mean
    xn = np.sqrt((x * x).sum())
    yn = np.sqrt((y * y).sum())
    if xn < 1e-8 or yn < 1e-8:
        return src - src_mean + dst_mean
    x0, y0 = x / xn, y / yn
    u, _, vt = np.linalg.svd(x0.T @ y0)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = u @ vt
    return (src - src_mean) @ r * (yn / xn) + dst_mean


def set_axes(ax, poses: list[np.ndarray]) -> None:
    pts = np.concatenate(poses, axis=0)
    center = pts.mean(axis=0)
    radius = max(float((pts.max(axis=0) - pts.min(axis=0)).max()) * 0.58, 0.2)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=13, azim=-78)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("X")
    ax.set_ylabel("depth")
    ax.set_zlabel("up")


def draw(ax, pose: np.ndarray, color: str, label: str, alpha: float = 1.0, lw: float = 2.8) -> None:
    for p, c in H36M_BONES:
        ax.plot(
            [pose[p, 0], pose[c, 0]],
            [pose[p, 1], pose[c, 1]],
            [pose[p, 2], pose[c, 2]],
            color=color,
            linewidth=lw,
            alpha=alpha,
        )
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=26, color=color, alpha=alpha, label=label)
    ax.scatter([pose[10, 0]], [pose[10, 1]], [pose[10, 2]], s=70, color=color, alpha=alpha)


def plot_one(path: Path, title: str, gt: np.ndarray, pred: np.ndarray, err: float) -> None:
    gt_o = orient_pose(gt)
    pred_o = similarity_align(orient_pose(pred), gt_o)

    fig = plt.figure(figsize=(12, 5.5), facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    draw(ax1, gt_o, "#1f77b4", "MM-Fi GT", 0.95, 3.0)
    set_axes(ax1, [gt_o])
    ax1.set_title("MM-Fi GT 17j", fontweight="bold")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw(ax2, gt_o, "#1f77b4", "GT", 0.45, 3.2)
    draw(ax2, pred_o, "#d62728", "Prediction", 0.95, 2.5)
    set_axes(ax2, [gt_o, pred_o])
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_title(f"Best model overlay | MPJPE={err:.1f}mm", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/ml/checkpoints/best.pt")
    p.add_argument("--gold-dir", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/data/gold/rfpose-mmfi-17j-v1")
    p.add_argument("--output-dir", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/viz_output/mmfi17j_best")
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--frame", type=int, default=30)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(Path(args.checkpoint), device)

    dataset = GoldNpzDataset(
        args.gold_dir,
        split="test",
        datasets=cfg["data"].get("datasets", ["mmfi"]),
        augment=False,
        require_pose=True,
        require_action=False,
    )

    rng = np.random.RandomState(7)
    selected = np.sort(rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False))
    results = []
    overview_items = []

    for n, idx in enumerate(selected):
        batch = dataset[int(idx)]
        csi = batch["csi"].unsqueeze(0).to(device)
        gt_seq = batch["coords"].numpy()
        with torch.no_grad():
            pred_seq = model(csi)["coords"].squeeze(0).cpu().numpy()
        f = min(args.frame, gt_seq.shape[0] - 1)
        gt = gt_seq[f]
        pred = pred_seq[f]
        err = float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)
        name = f"sample_{int(idx):04d}_frame_{f:02d}"
        out = out_dir / f"{name}.png"
        plot_one(out, name, gt, pred, err)
        print(f"saved {out} MPJPE={err:.1f}mm")
        results.append({"idx": int(idx), "frame": int(f), "mpjpe_mm": round(err, 1), "file": out.name})
        overview_items.append((name, orient_pose(gt), similarity_align(orient_pose(pred), orient_pose(gt)), err))

    cols = min(4, len(overview_items))
    rows = int(np.ceil(len(overview_items) / cols))
    fig = plt.figure(figsize=(4.7 * cols, 4.5 * rows), facecolor="white")
    for i, (name, gt, pred, err) in enumerate(overview_items, 1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        draw(ax, gt, "#1f77b4", "GT", 0.45, 2.7)
        draw(ax, pred, "#d62728", "Pred", 0.92, 2.2)
        set_axes(ax, [gt, pred])
        ax.set_title(f"{name}\n{err:.1f}mm", fontsize=9, fontweight="bold")
    fig.suptitle("MM-Fi 17j best model: GT (blue) vs prediction (red)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    overview = out_dir / "overview_mmfi17j_best.png"
    fig.savefig(overview, dpi=180)
    plt.close(fig)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"saved {overview}")


if __name__ == "__main__":
    main()
