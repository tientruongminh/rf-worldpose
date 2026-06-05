#!/usr/bin/env python3
"""Read and inspect Wi-Pose .mat files (MATLAB v7.3 / HDF5 format).

Usage:
    python tools/read_wipose_mat.py /mnt/d/Wi-Pose/Train/bend_001-frame001.mat
    python tools/read_wipose_mat.py /mnt/d/Wi-Pose/Train/bend_001-frame001.mat --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


NUM_JOINTS = 18
JOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
    "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar",
]


def load_mat(path: str | Path) -> dict[str, np.ndarray]:
    """Load a Wi-Pose .mat file and return CSI + skeleton arrays."""
    with h5py.File(str(path), "r") as f:
        return {key: f[key][()] for key in f.keys()}


def parse_skeleton(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parse the flat (1,54) skeleton into (18,2) coords and (18,) confidence."""
    flat = raw.flatten()
    xy = flat[: NUM_JOINTS * 2].reshape(NUM_JOINTS, 2)
    conf = flat[NUM_JOINTS * 2 :]
    return xy, conf


def print_summary(data: dict[str, np.ndarray]) -> None:
    for key, arr in data.items():
        print(f"\n{'=' * 50}")
        print(f"  {key}")
        print(f"{'=' * 50}")
        print(f"  shape : {arr.shape}")
        print(f"  dtype : {arr.dtype}")
        print(f"  min   : {arr.min():.6f}")
        print(f"  max   : {arr.max():.6f}")
        print(f"  mean  : {arr.mean():.6f}")

    if "CSI" in data:
        csi = data["CSI"]
        tx, rx, sub, pkt = csi.shape
        print(f"\n--- CSI breakdown ---")
        print(f"  Tx antennas   : {tx}")
        print(f"  Rx antennas   : {rx}")
        print(f"  Subcarriers   : {sub}")
        print(f"  Packets/steps : {pkt}")

    if "SkeletonPoints" in data:
        xy, conf = parse_skeleton(data["SkeletonPoints"])
        print(f"\n--- Skeleton joints (18 OpenPose keypoints) ---")
        print(f"  {'#':>2}  {'Joint':<12}  {'x':>8}  {'y':>8}  {'conf':>6}")
        print(f"  {'--':>2}  {'-----':<12}  {'---':>8}  {'---':>8}  {'----':>6}")
        for i in range(NUM_JOINTS):
            name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i}"
            print(f"  {i:2d}  {name:<12}  {xy[i, 0]:8.2f}  {xy[i, 1]:8.2f}  {conf[i]:6.2f}")


def plot(data: dict[str, np.ndarray], save_path: str | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- CSI heatmap (average over Tx/Rx pairs) ---
    csi = data["CSI"]
    csi_2d = csi.reshape(-1, csi.shape[2], csi.shape[3]).mean(axis=0)
    im = axes[0].imshow(csi_2d, aspect="auto", cmap="viridis")
    axes[0].set_title("CSI Amplitude (avg across Tx-Rx pairs)")
    axes[0].set_xlabel("Packet / Time step")
    axes[0].set_ylabel("Subcarrier index")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    # --- Skeleton plot ---
    BONES = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
        (0, 15), (0, 16), (15, 17),
    ]
    xy, conf = parse_skeleton(data["SkeletonPoints"])
    ax = axes[1]
    for a, b in BONES:
        if conf[a] > 0.1 and conf[b] > 0.1:
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], "b-", lw=1.5)
    visible = conf > 0.1
    ax.scatter(xy[visible, 0], xy[visible, 1], c="red", s=40, zorder=5)
    for i in range(NUM_JOINTS):
        if conf[i] > 0.1:
            name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i}"
            ax.annotate(name, (xy[i, 0], xy[i, 1]), fontsize=6, ha="center", va="bottom")
    ax.invert_yaxis()
    ax.set_title("Skeleton (OpenPose keypoints)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nPlot saved to {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Wi-Pose .mat files")
    parser.add_argument("mat_file", help="Path to .mat file")
    parser.add_argument("--plot", action="store_true", help="Show CSI + skeleton plot")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    path = Path(args.mat_file)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading: {path}")
    data = load_mat(path)
    print_summary(data)

    if args.plot or args.save_plot:
        plot(data, save_path=args.save_plot)


if __name__ == "__main__":
    main()
