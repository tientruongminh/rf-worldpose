#!/usr/bin/env python3
"""Compare GT vs 3 WiMose checkpoints on MM-Fi test set.

Visual polish for human-like poses:
  1. Retarget each prediction to GT bone lengths (per sample)
  2. Procrustes-align prediction to GT for overlay display
  3. Pick samples with lowest PA-MPJPE after retarget (best-looking overlays)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rfpose.data.gold_npz_dataset import GoldNpzDataset
from rfpose.models.wimose_net import WiMoseNet

H36M_PARENT = {
    1: 0, 2: 1, 3: 2,
    4: 0, 5: 4, 6: 5,
    7: 0, 8: 7, 9: 8, 10: 9,
    11: 8, 12: 11, 13: 12,
    14: 8, 15: 14, 16: 15,
}
BONES = [(p, c) for c, p in H36M_PARENT.items()]


@dataclass
class ModelSpec:
    name: str
    checkpoint: Path
    use_gcn_head: bool = False


def load_model(spec: ModelSpec, device: torch.device) -> tuple[WiMoseNet, dict]:
    ckpt = torch.load(spec.checkpoint, map_location=device, weights_only=False)
    model = WiMoseNet(
        n_joints=int(ckpt.get("n_joints", 17)),
        in_channels=2,
        use_gcn_head=spec.use_gcn_head,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def prepare_csi(batch: dict, ckpt: dict, device: torch.device) -> torch.Tensor:
    csi = batch["csi"].unsqueeze(0).to(device)
    x = csi.permute(0, 3, 2, 1).contiguous()
    mean, std = ckpt.get("csi_mean"), ckpt.get("csi_std")
    if mean is not None and std is not None:
        x = (x - mean.to(device)) / std.to(device)
    return x


def midpoint_gt(batch: dict, ckpt: dict) -> np.ndarray:
    gt = batch["coords"].numpy()[batch["coords"].shape[0] // 2].astype(np.float32)
    if bool(ckpt.get("center_pose", True)):
        root = int(ckpt.get("root_joint", 0))
        if 0 <= root < gt.shape[0]:
            gt = gt - gt[root : root + 1]
    return gt


def bone_lengths(pose: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(pose[c] - pose[p]) for p, c in BONES], dtype=np.float32)


def retarget_to_lengths(pred: np.ndarray, target_lengths: np.ndarray, root: int) -> np.ndarray:
    children: dict[int, list[tuple[int, int]]] = {}
    for i, (p, c) in enumerate(BONES):
        children.setdefault(p, []).append((c, i))
    out = pred.copy()
    out[root] = pred[root]
    stack = [root]
    seen = {root}
    while stack:
        par = stack.pop()
        for child, bone_idx in children.get(par, []):
            vec = pred[child] - pred[par]
            ln = float(np.linalg.norm(vec))
            out[child] = out[par] if ln < 1e-6 else out[par] + vec / ln * float(target_lengths[bone_idx])
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return out


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity transform: scale + rotation (no translation; both root-centered)."""
    h = pred.T @ gt
    u, s, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    scale = s.sum() / max(float((pred ** 2).sum()), 1e-8)
    return scale * pred @ r.T


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(procrustes_align(pred, gt) - gt, axis=-1).mean())


def raw_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def humanize(pred: np.ndarray, gt: np.ndarray, root: int) -> np.ndarray:
    """GT bone lengths + Procrustes for display."""
    bonelen = retarget_to_lengths(pred, bone_lengths(gt), root)
    return procrustes_align(bonelen, gt)


def xyz(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pose[:, 0], pose[:, 2], -pose[:, 1]


def bone_color(u: int, v: int) -> str:
    right = {1, 2, 3, 14, 15, 16}
    left = {4, 5, 6, 11, 12, 13}
    if u in right or v in right:
        return "#ff7f0e"
    if u in left or v in left:
        return "#1f77b4"
    return "#2ca02c"


def draw_skel(ax, pose: np.ndarray, color_fn, alpha: float, lw: float, s: float) -> None:
    x, y, z = xyz(pose)
    cols = [color_fn(i, i) for i in range(len(pose))] if color_fn else ["#d62728"] * len(pose)
    ax.scatter(x, y, z, s=s, c=cols, edgecolors="k", linewidths=0.35, alpha=alpha, zorder=5)
    for p, c in BONES:
        col = color_fn(p, c) if color_fn else "#d62728"
        ax.plot([x[p], x[c]], [y[p], y[c]], [z[p], z[c]], color=col, lw=lw, alpha=alpha)


def set_axes(ax, poses: list[np.ndarray]) -> None:
    pts = np.concatenate(poses, axis=0)
    x, y, z = xyz(pts)
    cx, cy, cz = x.mean(), y.mean(), z.mean()
    r = max(float(np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()) * 0.55, 0.25)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    ax.view_init(elev=12, azim=-70)
    ax.grid(True, linestyle="--", alpha=0.15)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("depth", fontsize=7)
    ax.set_zlabel("up", fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proto1-ckpt", required=True)
    parser.add_argument("--cross-ckpt", required=True)
    parser.add_argument("--unif-ckpt", required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--scan-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = [
        ModelSpec("Proto1 MLP", Path(args.proto1_ckpt), use_gcn_head=False),
        ModelSpec("Cross-subj MLP", Path(args.cross_ckpt), use_gcn_head=False),
        ModelSpec("GCN+Unif+JW", Path(args.unif_ckpt), use_gcn_head=True),
    ]
    models = [(s, *load_model(s, device)) for s in specs]

    ds = GoldNpzDataset(
        args.gold_dir,
        split="test",
        datasets=["mmfi"],
        augment=False,
        require_pose=True,
    )
    rng = np.random.RandomState(args.seed)
    scan_n = min(args.scan_samples, len(ds))
    indices = rng.choice(len(ds), size=scan_n, replace=False)

    scored: list[dict] = []
    for idx in indices:
        batch = ds[int(idx)]
        gt = midpoint_gt(batch, models[0][2])  # same root centering for all
        root = int(models[0][2].get("root_joint", 0))
        rec = {"idx": int(idx), "gt": gt, "preds": {}, "metrics": {}}
        avg_pa = 0.0
        for spec, model, ckpt in models:
            with torch.no_grad():
                raw = model(prepare_csi(batch, ckpt, device)).squeeze(0).cpu().numpy().astype(np.float32)
            viz = humanize(raw, gt, root)
            rec["preds"][spec.name] = viz
            rec["metrics"][spec.name] = {
                "raw_mm": raw_mpjpe(raw, gt) * 1000,
                "pa_mm": pa_mpjpe(raw, gt) * 1000,
            }
            avg_pa += rec["metrics"][spec.name]["pa_mm"]
        rec["avg_pa_mm"] = avg_pa / len(models)
        meta = batch.get("meta", {})
        rec["sample_id"] = str(meta.get("sample_id", f"test_{int(idx):04d}"))
        scored.append(rec)

    # Best-looking: lowest average PA-MPJPE across all 3 models
    scored.sort(key=lambda r: r["avg_pa_mm"])
    best = scored[: args.num_samples]

    # ── Figure 1: overlay GT + all 3 preds per sample ───────────────────────
    cols = args.num_samples
    fig = plt.figure(figsize=(5.5 * cols, 16), facecolor="white")
    model_names = [s.name for s in specs]
    for col, rec in enumerate(best, 1):
        gt = rec["gt"]
        for row, mname in enumerate(["GT"] + model_names, 1):
            ax = fig.add_subplot(4, cols, (row - 1) * cols + col, projection="3d")
            if mname == "GT":
                draw_skel(ax, gt, bone_color, alpha=0.85, lw=2.8, s=45)
                title = f"{rec['sample_id']}\nGT"
            else:
                pred = rec["preds"][mname]
                draw_skel(ax, gt, bone_color, alpha=0.35, lw=1.8, s=28)
                draw_skel(ax, pred, None, alpha=0.95, lw=2.2, s=30)
                m = rec["metrics"][mname]
                title = f"{mname}\nraw={m['raw_mm']:.0f} PA={m['pa_mm']:.0f}mm"
            set_axes(ax, [gt, rec["preds"][model_names[0]]])
            ax.set_title(title, fontsize=8, fontweight="bold")
    fig.suptitle(
        "MM-Fi test | GT bone-length + Procrustes | Best 6 by avg PA-MPJPE",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    overlay_path = out_dir / "compare_3model_overlay.png"
    fig.savefig(overlay_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: side-by-side GT vs each model (cleaner) ───────────────────
    fig2, axes = plt.subplots(
        len(best), 4, figsize=(18, 3.8 * len(best)),
        subplot_kw={"projection": "3d"},
        facecolor="white",
    )
    if len(best) == 1:
        axes = np.array([axes])
    headers = ["GT", "Proto1 MLP\n157.7mm", "Cross-subj MLP\n169.3mm", "GCN+Unif+JW\n174.3mm"]
    for row, rec in enumerate(best):
        panels = [rec["gt"], rec["preds"][model_names[0]], rec["preds"][model_names[1]], rec["preds"][model_names[2]]]
        for col, (ax, pose, hdr) in enumerate(zip(axes[row], panels, headers)):
            if col == 0:
                draw_skel(ax, pose, bone_color, alpha=0.9, lw=3.0, s=50)
            else:
                draw_skel(ax, rec["gt"], bone_color, alpha=0.3, lw=1.5, s=22)
                draw_skel(ax, pose, None, alpha=0.95, lw=2.4, s=32)
            set_axes(ax, [rec["gt"], pose])
            if row == 0:
                ax.set_title(hdr, fontsize=9, fontweight="bold")
            if col == 0:
                ax.text2D(0.02, 0.92, rec["sample_id"], transform=ax.transAxes, fontsize=8)
            elif col > 0:
                m = rec["metrics"][model_names[col - 1]]
                ax.text2D(0.02, 0.92, f"PA={m['pa_mm']:.0f}mm", transform=ax.transAxes, fontsize=8, color="#d62728")
    fig2.suptitle("GT vs 3 models | bone-length normalized + Procrustes aligned", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = out_dir / "compare_3model_grid.png"
    fig2.savefig(grid_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "output_dir": str(out_dir),
        "samples": [
            {
                "sample_id": r["sample_id"],
                "idx": r["idx"],
                "avg_pa_mm": round(r["avg_pa_mm"], 1),
                "metrics": r["metrics"],
            }
            for r in best
        ],
        "mean_pa_mm": {
            name: round(float(np.mean([r["metrics"][name]["pa_mm"] for r in best])), 1)
            for name in model_names
        },
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved: {overlay_path}")
    print(f"saved: {grid_path}")


if __name__ == "__main__":
    main()
