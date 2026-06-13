#!/usr/bin/env python3
"""MM-Fi prediction-vs-GT visualization in the same style as EDA bronze plots."""

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


MMFI_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8),
    (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def get_bone_color(u: int, v: int) -> str:
    right_side = {1, 2, 3, 14, 15, 16}
    left_side = {4, 5, 6, 11, 12, 13}
    if u in right_side or v in right_side:
        return "#ff7f0e"
    if u in left_side or v in left_side:
        return "#1f77b4"
    return "#2ca02c"


ACTION_MAPPING = {
    "A01": "Stretching", "A02": "Chest Exp (H)", "A03": "Chest Exp (V)",
    "A04": "Twist (L)", "A05": "Twist (R)", "A06": "Mark Time",
    "A07": "Limb Ext (L)", "A08": "Limb Ext (R)", "A09": "Lunge (LF)",
    "A10": "Lunge (RF)", "A11": "Limb Ext (Both)", "A12": "Squat",
    "A13": "Raising Hand (L)", "A14": "Raising Hand (R)",
    "A17": "Waving Hand (L)", "A19": "Picking Up",
    "A20": "Throwing (L)", "A26": "Jumping Up",
}


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


def xyz_for_plot(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Exact EDA convention: X=width, Y=depth, Z=height.
    return points[:, 0], points[:, 2], -points[:, 1]


def draw_gt(ax, pts: np.ndarray) -> None:
    x, y, z = xyz_for_plot(pts)
    node_colors = plt.cm.tab20(np.linspace(0, 1, 17))
    ax.scatter(x, y, z, c=node_colors, s=45, zorder=5, edgecolors="k", linewidths=0.5, alpha=0.95)
    for u, v in MMFI_SKELETON_CONNECTIONS:
        color = get_bone_color(u, v)
        ax.plot([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], color=color, linewidth=2.5, alpha=0.78, zorder=3)


def draw_pred(ax, pts: np.ndarray) -> None:
    x, y, z = xyz_for_plot(pts)
    ax.scatter(x, y, z, c="#d62728", s=34, zorder=7, edgecolors="white", linewidths=0.4, alpha=0.90)
    for u, v in MMFI_SKELETON_CONNECTIONS:
        ax.plot([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], color="#d62728", linewidth=2.0, alpha=0.88, zorder=6)


def set_eda_axes(ax, gt_seq: np.ndarray, pred_seq: np.ndarray | None = None) -> None:
    seqs = [gt_seq]
    if pred_seq is not None:
        seqs.append(pred_seq)
    all_pts = np.concatenate(seqs, axis=0)
    x_all = all_pts[:, :, 0]
    y_all = all_pts[:, :, 2]
    z_all = -all_pts[:, :, 1]
    mid_x = (x_all.max() + x_all.min()) / 2.0
    mid_y = (y_all.max() + y_all.min()) / 2.0
    mid_z = (z_all.max() + z_all.min()) / 2.0
    max_range = np.array([x_all.max() - x_all.min(), y_all.max() - y_all.min(), z_all.max() - z_all.min()]).max() / 2.0
    max_range = max(float(max_range), 1e-3)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel("X (Width)", fontsize=8)
    ax.set_ylabel("Y (Depth)", fontsize=8)
    ax.set_zlabel("Z (Height)", fontsize=8)
    ax.view_init(elev=15, azim=-65)
    ax.grid(True, linestyle="--", alpha=0.2)


def sample_title(meta: dict) -> str:
    sample_id = meta.get("sample_id", "sample")
    action = sample_id.split("_")[-1] if "_" in sample_id else sample_id
    action_name = ACTION_MAPPING.get(action, action)
    return f"{sample_id} {action_name}"


def plot_progression(out_path: Path, meta: dict, gt_seq: np.ndarray, pred_seq: np.ndarray) -> list[dict]:
    t_len = gt_seq.shape[0]
    frame_indices = np.linspace(0, t_len - 1, 6, dtype=int)
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")
    records = []
    for sub_idx, f_idx in enumerate(frame_indices):
        gt = gt_seq[f_idx]
        pred = pred_seq[f_idx]
        err = float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)
        records.append({"frame": int(f_idx), "mpjpe_mm": round(err, 1)})

        ax = fig.add_subplot(2, 3, sub_idx + 1, projection="3d")
        ax.set_facecolor("white")
        draw_gt(ax, gt)
        draw_pred(ax, pred)
        set_eda_axes(ax, gt_seq, pred_seq)
        progress_pct = int((f_idx / max(t_len - 1, 1)) * 100)
        ax.set_title(f"Frame {f_idx} ({progress_pct}%) | {err:.1f}mm", fontsize=11, fontweight="bold", color="#d62728")

    fig.suptitle(
        f"MM-Fi Predict vs GT - EDA style\n{sample_title(meta)} | GT=color skeleton, Pred=red",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/ml/checkpoints/best.pt")
    parser.add_argument("--gold-dir", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/data/gold/rfpose-mmfi-17j-v1")
    parser.add_argument("--output-dir", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/viz_output/mmfi17j_eda_overlay")
    parser.add_argument("--indices", nargs="*", type=int, default=[])
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(Path(args.checkpoint), device)
    dataset = GoldNpzDataset(
        args.gold_dir,
        split=None,
        datasets=cfg["data"].get("datasets", ["mmfi"]),
        augment=False,
        require_pose=True,
        require_action=False,
    )
    meta = np.load(Path(args.gold_dir) / "mmfi" / "metadata.npz", allow_pickle=True)["metadata"]

    if args.indices:
        indices = args.indices[: args.num_samples]
    else:
        # Pick samples that correspond to existing bronze EDA actions where possible.
        wanted = {"A06", "A12", "A19", "A22", "A23", "A24", "A25", "A26"}
        indices = []
        seen = set()
        for i, m in enumerate(meta):
            action = m["sample_id"].split("_")[-1]
            if action in wanted and action not in seen:
                indices.append(i)
                seen.add(action)
            if len(indices) >= args.num_samples:
                break
    results = []
    for idx in indices:
        batch = dataset[idx]
        with torch.no_grad():
            pred_seq = model(batch["csi"].unsqueeze(0).to(device))["coords"].squeeze(0).cpu().numpy()
        gt_seq = batch["coords"].numpy()
        m = meta[idx]
        out_path = out_dir / f"eda_overlay_{idx:04d}_{m['sample_id']}.png"
        records = plot_progression(out_path, m, gt_seq, pred_seq)
        print(f"saved {out_path}")
        results.append({"idx": int(idx), "metadata": dict(m), "file": out_path.name, "frames": records})
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
