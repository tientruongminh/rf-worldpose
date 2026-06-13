#!/usr/bin/env python3
"""Render MM-Fi 17j Gold ground-truth skeleton samples for sanity checking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


H36M_BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

BODY_COLORS = {
    "spine": "#2f6fed",
    "right": "#f28e2b",
    "left": "#59a14f",
    "head": "#af7aa1",
}


def bone_color(a: int, b: int) -> str:
    pair = {a, b}
    if pair <= {0, 7, 8, 9, 10}:
        return BODY_COLORS["spine"] if 10 not in pair else BODY_COLORS["head"]
    if a in {1, 2, 3, 14, 15, 16} or b in {1, 2, 3, 14, 15, 16}:
        return BODY_COLORS["right"]
    return BODY_COLORS["left"]


def orient_eda(pts: np.ndarray) -> np.ndarray:
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = pts[:, 2]
    out[:, 2] = -pts[:, 1]
    return out


def set_equal_axes(ax, pts: np.ndarray) -> None:
    center = pts.mean(axis=0)
    span = np.ptp(pts, axis=0).max()
    radius = max(float(span) * 0.65, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_pose(ax, pts: np.ndarray, title: str) -> None:
    pts = orient_eda(pts)
    for a, b in H36M_BONES:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            [pts[a, 2], pts[b, 2]],
            color=bone_color(a, b),
            linewidth=3,
        )
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#222222", s=18, depthshade=True)
    set_equal_axes(ax, pts)
    ax.view_init(elev=15, azim=-65)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold-dir", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/data/gold/rfpose-mmfi-17j-v1/mmfi")
    p.add_argument("--out-dir", default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/viz_output/mmfi_gold_check")
    p.add_argument("--per-split", type=int, default=4)
    p.add_argument("--frame", type=int, default=30)
    args = p.parse_args()

    gold = Path(args.gold_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    y = np.load(gold / "y.npz")["pose"]
    meta = np.load(gold / "metadata.npz", allow_pickle=True)["metadata"]

    chosen = []
    for split in ["train", "val", "test"]:
        idxs = [i for i, m in enumerate(meta) if m.get("split") == split]
        if not idxs:
            continue
        step = max(1, len(idxs) // args.per_split)
        chosen.extend(idxs[::step][: args.per_split])

    summary = []
    for idx in chosen:
        m = meta[idx]
        frame = min(args.frame, y.shape[1] - 1)
        pts = y[idx, frame]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        title = f"MM-Fi Gold GT | {m.get('split')} | {m.get('sample_id')} | idx={idx} f={frame}"
        draw_pose(ax, pts, title)
        fig.tight_layout()
        out_path = out / f"mmfi_gold_{m.get('split')}_{idx:05d}_{m.get('sample_id')}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        summary.append({
            "idx": int(idx),
            "split": m.get("split"),
            "sample_id": m.get("sample_id"),
            "window_start": int(m.get("window_start", 0)),
            "frame": int(frame),
            "path": str(out_path),
            "range": [float(pts.min()), float(pts.max())],
        })

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"out_dir": str(out), "num_images": len(summary), "images": [s["path"] for s in summary]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
