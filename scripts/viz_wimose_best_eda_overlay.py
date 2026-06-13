#!/usr/bin/env python3
"""Visualize WiMose best checkpoints as GT-vs-pred 3D skeleton overlays."""

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
from rfpose.models.wimose_net import WiMoseNet


BONES = {
    "mmfi": [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16),
    ],
    "wipose": [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10),
        (1, 11), (11, 12), (12, 13),
        (8, 11), (0, 14), (14, 16), (0, 15), (15, 17),
    ],
}


def load_model(checkpoint: Path, dataset: str, device: torch.device) -> tuple[WiMoseNet, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    n_joints = int(ckpt.get("n_joints", 17 if dataset == "mmfi" else 18))
    model = WiMoseNet(n_joints=n_joints, in_channels=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def prepare_csi(batch: dict, ckpt: dict, device: torch.device) -> torch.Tensor:
    csi = batch["csi"].unsqueeze(0).to(device)  # [1,T,N,2]
    x = csi.permute(0, 3, 2, 1).contiguous()    # [1,2,N,T]
    mean = ckpt.get("csi_mean")
    std = ckpt.get("csi_std")
    if mean is not None and std is not None:
        x = (x - mean.to(device)) / std.to(device)
    return x


def midpoint_gt(batch: dict, ckpt: dict) -> np.ndarray:
    coords = batch["coords"].numpy()
    gt = coords[coords.shape[0] // 2].astype(np.float32)
    if bool(ckpt.get("center_pose", True)):
        root = int(ckpt.get("root_joint", 0))
        if 0 <= root < gt.shape[0]:
            gt = gt - gt[root : root + 1]
    return gt


def canonical_bone_lengths(gold_dir: str, dataset: str, bones: list[tuple[int, int]], ckpt: dict, max_samples: int = 1024) -> np.ndarray:
    ds = GoldNpzDataset(
        gold_dir,
        split="train",
        datasets=[dataset],
        augment=False,
        require_pose=True,
        require_action=False,
    )
    rng = np.random.RandomState(123)
    indices = rng.choice(len(ds), size=min(max_samples, len(ds)), replace=False)
    root = int(ckpt.get("root_joint", 0))
    lengths = []
    for idx in indices:
        batch = ds[int(idx)]
        gt = batch["coords"].numpy()[batch["coords"].shape[0] // 2].astype(np.float32)
        if bool(ckpt.get("center_pose", True)) and 0 <= root < gt.shape[0]:
            gt = gt - gt[root : root + 1]
        lengths.append([np.linalg.norm(gt[v] - gt[u]) for u, v in bones])
    return np.median(np.asarray(lengths, dtype=np.float32), axis=0)


def retarget_bone_lengths(pred: np.ndarray, bones: list[tuple[int, int]], target_lengths: np.ndarray, root: int) -> np.ndarray:
    children: dict[int, list[tuple[int, int]]] = {}
    for i, (u, v) in enumerate(bones):
        children.setdefault(u, []).append((v, i))
    out = pred.astype(np.float32).copy()
    out[root] = pred[root]
    stack = [root]
    visited = {root}
    while stack:
        u = stack.pop()
        for v, bone_idx in children.get(u, []):
            vec = pred[v] - pred[u]
            norm = float(np.linalg.norm(vec))
            if norm < 1e-6:
                out[v] = pred[v]
            else:
                out[v] = out[u] + vec / norm * float(target_lengths[bone_idx])
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return out


def xyz_for_plot(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return points[:, 0], points[:, 2], -points[:, 1]


def bone_color(u: int, v: int) -> str:
    right = {1, 2, 3, 14, 15, 16}
    left = {4, 5, 6, 11, 12, 13}
    if u in right or v in right:
        return "#ff7f0e"
    if u in left or v in left:
        return "#1f77b4"
    return "#2ca02c"


def draw_gt(ax, pose: np.ndarray, bones: list[tuple[int, int]]) -> None:
    x, y, z = xyz_for_plot(pose)
    ax.scatter(x, y, z, c=plt.cm.tab20(np.linspace(0, 1, len(pose))), s=42, edgecolors="k", linewidths=0.4)
    for u, v in bones:
        ax.plot([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], color=bone_color(u, v), linewidth=2.5, alpha=0.75)


def draw_pred(ax, pose: np.ndarray, bones: list[tuple[int, int]]) -> None:
    x, y, z = xyz_for_plot(pose)
    ax.scatter(x, y, z, c="#d62728", s=34, edgecolors="white", linewidths=0.4, alpha=0.95)
    for u, v in bones:
        ax.plot([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], color="#d62728", linewidth=2.0, alpha=0.9)


def set_axes(ax, poses: list[np.ndarray]) -> None:
    pts = np.concatenate(poses, axis=0)
    x, y, z = xyz_for_plot(pts)
    mid = np.array([(x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2])
    rad = max(float(np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()) * 0.58, 1e-3)
    ax.set_xlim(mid[0] - rad, mid[0] + rad)
    ax.set_ylim(mid[1] - rad, mid[1] + rad)
    ax.set_zlim(mid[2] - rad, mid[2] + rad)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y/depth", fontsize=8)
    ax.set_zlabel("Z/up", fontsize=8)
    ax.view_init(elev=15, azim=-65)
    ax.grid(True, linestyle="--", alpha=0.18)


def plot_grid(out_path: Path, title: str, items: list[dict], bones: list[tuple[int, int]]) -> None:
    cols = min(3, len(items))
    rows = int(np.ceil(len(items) / cols))
    fig = plt.figure(figsize=(6.0 * cols, 5.2 * rows), facecolor="white")
    for i, rec in enumerate(items, 1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        draw_gt(ax, rec["gt"], bones)
        draw_pred(ax, rec["pred"], bones)
        set_axes(ax, [rec["gt"], rec["pred"]])
        ax.set_title(f"{rec['sample_id']} | {rec['mpjpe']:.4f}", fontsize=10, fontweight="bold")
    fig.suptitle(title + " | GT=color, Pred=red", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--dataset", choices=["mmfi", "wipose"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--retarget-canonical", action="store_true")
    parser.add_argument("--canonical-samples", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(Path(args.checkpoint), args.dataset, device)
    bones = BONES[args.dataset]
    target_lengths = None
    if args.retarget_canonical:
        target_lengths = canonical_bone_lengths(args.gold_dir, args.dataset, bones, ckpt, max_samples=args.canonical_samples)
        print("canonical_bone_lengths", target_lengths.tolist())

    all_results = []
    for split in args.splits:
        ds = GoldNpzDataset(
            args.gold_dir,
            split=split,
            datasets=[args.dataset],
            augment=False,
            require_pose=True,
            require_action=False,
        )
        rng = np.random.RandomState(args.seed + sum(ord(c) for c in split))
        indices = np.sort(rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False))
        items = []
        for idx in indices:
            batch = ds[int(idx)]
            with torch.no_grad():
                pred = model(prepare_csi(batch, ckpt, device)).squeeze(0).cpu().numpy().astype(np.float32)
            gt = midpoint_gt(batch, ckpt)
            if target_lengths is not None:
                pred = retarget_bone_lengths(pred, bones, target_lengths, int(ckpt.get("root_joint", 0)))
            mpjpe = float(np.linalg.norm(pred - gt, axis=-1).mean())
            meta = batch.get("meta", {})
            sample_id = str(meta.get("sample_id", f"{split}_{int(idx):04d}"))
            rec = {"split": split, "idx": int(idx), "sample_id": sample_id, "mpjpe": mpjpe}
            items.append({**rec, "gt": gt, "pred": pred})
            all_results.append(rec)
        plot_grid(out_dir / f"{args.dataset}_{split}_wimose_best_overlay.png", f"WiMose {args.dataset} {split} best epoch", items, bones)

    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    print(json.dumps({"output_dir": str(out_dir), "results": all_results}, indent=2))


if __name__ == "__main__":
    main()
