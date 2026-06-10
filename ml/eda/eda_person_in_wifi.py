#!/usr/bin/env python3
"""
EDA - Person-in-WiFi 3D (wipose_dataset)

Gồm 3 Figures: (piw_dataset_overview, piw_keypoint_geometry, piw_phase_denoising)

Đường dẫn từ configs.py.
Result saved: /results_person_in_wifi
"""

import os
import glob
import warnings
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import configs 

warnings.filterwarnings("ignore")

DATASET_DIR = configs.PIW_DATA_DIR
OUT_DIR     = configs.PIW_RESULTS_DIR

TRAIN_DIR   = os.path.join(DATASET_DIR, "train_data")
CSI_DIR     = os.path.join(TRAIN_DIR, "csi")
KP_DIR      = os.path.join(TRAIN_DIR, "keypoint")

os.makedirs(OUT_DIR, exist_ok=True)

JOINT_NAMES = [
    "neck", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_hip", "r_hip", "l_hand", "r_hand", "l_knee", "r_knee", "l_ankle", "r_ankle"
]

STYLE = {
    "fig_bg":  "#FAFAF8",
    "axes_bg": "#F5F4F0",
    "text":    "#2C2C2A",
    "grid":    "#DDDBD0",
    # ── 💡 ĐỒNG BỘ MÃ MÀU PALETTE MỊN MÀU THEO ẢNH MẪU ──
    "accent1": "#4C72B0",  # Xanh dương Steel Blue (Cột X)
    "accent2": "#55A868",  # Xanh ngọc Muted Green (Cột Y)
    "accent3": "#C44E52",  # Cam đất / Đỏ gạch Muted Red (Cột Z)
    "paper_colors": ["#4C72B0", "#55A868", "#C44E52"]  # Đồng bộ dải màu cho Fig 3
}

def load_mat(path):
    with h5py.File(path, 'r') as f:
        data_keys = [k for k in f.keys() if not k.startswith("_")]
        if not data_keys:
            raise ValueError(f"No valid data keys found in HDF5/MAT file: {path}")
        csi_candidates = [k for k in data_keys if "csi" in k.lower()]
        key = csi_candidates[0] if len(csi_candidates) > 0 else data_keys[0]
        data = f[key]
        if isinstance(data, h5py.Dataset) and ('real' in data.dtype.names and 'imag' in data.dtype.names):
            arr = np.array(data['real']) + np.array(data['imag']) * 1j
        else:
            arr = np.array(data)

    arr = np.array(arr)
    if arr.ndim == 4:
        arr = arr.transpose(3, 2, 1, 0)
        arr = arr[np.newaxis, np.newaxis, ...]
    elif arr.ndim == 5:
        arr = arr.transpose(4, 3, 2, 1, 0)
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 6:
        arr = arr.transpose(5, 4, 3, 2, 1, 0)
    return arr, key

def load_npy(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 3:
        arr = arr[:, np.newaxis, :, :]
    return arr

def linear_phase_clean(csi_vector):
    """
    Replicates the exact Linear Transformation method
    Removes time-frequency synchronization errors across 30 subcarriers.
    """
    pha_raw = np.angle(csi_vector)
    pha_unwrap = np.unwrap(pha_raw)
    
    S = len(pha_unwrap) # 30 Subcarriers
    subcarrier_idx = np.arange(S, dtype=np.float64)
    
    idx_mean = subcarrier_idx.mean()
    idx_centered = subcarrier_idx - idx_mean
    
    slope = np.sum(pha_unwrap * idx_centered) / np.sum(idx_centered ** 2)
    intercept = pha_unwrap.mean()
    
    phase_noise_trend = slope * idx_centered + intercept
    pha_sanitized = pha_unwrap - phase_noise_trend
    return pha_raw, pha_sanitized

def scan_dataset():
    print("\n[INFO] Scanning dataset directory...", flush=True)
    mat_files = sorted(glob.glob(os.path.join(CSI_DIR, "*.mat")))
    npy_files = sorted(glob.glob(os.path.join(KP_DIR, "*.npy")))
    mat_stems = {Path(f).stem for f in mat_files}
    npy_stems = {Path(f).stem for f in npy_files}
    common    = sorted(mat_stems & npy_stems)
    
    meta = []
    for stem in common:
        meta.append({
            "stem": stem,
            "csi_path": os.path.join(CSI_DIR, stem + ".mat"),
            "kp_path": os.path.join(KP_DIR, stem + ".npy")
        })
    print(f"[SUCCESS] Found {len(meta)} synchronized paired samples.", flush=True)
    return meta, mat_files, npy_files

# =====================================================================
# FIGURE 1: DATASET OVERVIEW (piw_dataset_overview.png)
# =====================================================================
def fig1_overview(meta, mat_files, npy_files):
    print("[RUN] Generating Figure 1: Dataset Overview...", flush=True)
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor(STYLE["fig_bg"])
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # File repository statistics bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(STYLE["axes_bg"])
    categories = ["CSI (.mat)", "KP (.npy)", "Synchronized"]
    counts = [len(mat_files), len(npy_files), len(meta)]

    bars = ax1.bar(categories, counts, color=[STYLE["accent1"], STYLE["accent2"], STYLE["accent3"]], width=0.5, zorder=3)
    
    for b, v in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("File Count")
    ax1.set_title("Dataset File Repository Statistics")
    ax1.grid(color=STYLE["grid"], linewidth=0.5, linestyle="--", alpha=0.5)

    # Sample distribution pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    train_kp = {"1-Person": 28121, "2-Person": 36242, "3-Person": 25583}
    test_kp  = {"1-Person": 2586,  "2-Person": 3184,  "3-Person": 2054}
    labels = list(train_kp.keys())
    sizes = [train_kp[l] + test_kp[l] for l in labels]
    ax2.pie(sizes, labels=labels, autopct="%1.1f%%", colors=[STYLE["accent1"], STYLE["accent2"], STYLE["accent3"]],
            startangle=140, textprops={"fontsize": 9, "color": STYLE["text"]}, wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
    ax2.set_title("Sample Distribution by Multi-Person Scenarios")

    # Data structural descriptions text panel
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    info_lines = [
        "Matrix Structural Dimensions (Shape Summary):",
        "──────────────────────────────────────────────",
        "Raw Wi-Fi CSI (.mat) Dimension:",
        "  Shape: (1, 3, 3, 30, 20)",
        "  Layout: (TX, RX, Ant, Subcarrier, Time)",
        "",
        "3D Skeleton Keypoints (.npy) Dimension:",
        "  Shape: (N, P, 14, 3)",
        "  Layout: (Frame, Person, Joint, Coordinate X/Y/Z)",
        "",
        "Spatial Field Environment Constraints:",
        "  Rectangle Testing Area: 4000mm x 3500mm",
        "  Ground Truth Source: Azure Kinect Body SDK"
    ]
    ax3.text(0.05, 0.95, "\n".join(info_lines), transform=ax3.transAxes,
             va="top", fontsize=9, color=STYLE["text"], fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor=STYLE["axes_bg"], edgecolor=STYLE["grid"]))
    ax3.set_title("Data Structural Descriptions")

    out = os.path.join(OUT_DIR, "piw_dataset_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=STYLE["fig_bg"])
    plt.close(fig)
    print(f"[SAVE] Figure 1: {out}", flush=True)

# =====================================================================
# FIGURE 2: KEYPOINT GEOMETRY ANALYSIS (piw_keypoint_geometry.png)
# =====================================================================
def fig2_keypoint_geometry(meta):
    print("[RUN] Generating Figure 2: Keypoint Spatial Geometry Analysis...", flush=True)
    all_kp = []
    for info in meta[:10]:
        try:
            kp = load_npy(info["kp_path"])
            valid = ~((kp == 0).all(axis=(-2, -1)))
            if kp[valid].shape[0] > 0:
                all_kp.append(kp[valid])
        except Exception: pass
        
    if not all_kp: return
    kp_pool = np.concatenate(all_kp, axis=0)

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor(STYLE["fig_bg"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    dim_names  = ["X — Horizontal Width (mm)", "Y — Depth Distance (mm)", "Z — Vertical Height (mm)"]
    dim_colors = [STYLE["accent1"], STYLE["accent2"], STYLE["accent3"]]

    for d in range(3):
        ax = fig.add_subplot(gs[0, d])
        ax.set_facecolor(STYLE["axes_bg"])
        axis_letter = ['X', 'Y', 'Z'][d]
        vals = kp_pool[:, :, d].flatten()
        
        ax.hist(vals, bins=40, color=dim_colors[d], alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
        ax.axvline(np.median(vals), color="black", linestyle="--", linewidth=1.2, label=f"median={np.median(vals):.0f}")
        ax.set_xlabel(dim_names[d])
        ax.set_ylabel("Sample Density / Count")
        ax.set_title(f"Spatial Distribution - Axis {axis_letter}")
        ax.grid(color=STYLE["grid"], linewidth=0.5, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    ax_range = fig.add_subplot(gs[1, :])
    ax_range.set_facecolor(STYLE["axes_bg"])
    x_pos = np.arange(14)
    for d, (ds, dc) in enumerate(zip(["X Axis (Width)", "Y Axis (Depth)", "Z Axis (Height)"], dim_colors)):
        max_vals = np.max(kp_pool[:, :, d], axis=0)
        min_vals = np.min(kp_pool[:, :, d], axis=0)
        ranges = max_vals - min_vals
        
        ax_range.bar(x_pos + d*0.23, ranges, width=0.23, color=dc, alpha=0.9, label=ds, zorder=3)
    
    ax_range.set_xticks(x_pos + 0.23)
    ax_range.set_xticklabels(JOINT_NAMES, rotation=30, ha="right", fontsize=8.5)
    ax_range.set_ylabel("Maximum Spatial Displacement Range (mm)")
    ax_range.set_title("Bi-directional Joint Movement Extent")
    ax_range.grid(color=STYLE["grid"], linewidth=0.5, linestyle="--", alpha=0.5)
    ax_range.legend(fontsize=9)

    out = os.path.join(OUT_DIR, "piw_keypoint_geometry.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=STYLE["fig_bg"])
    plt.close(fig)
    print(f"[SAVE] Figure 2: {out}", flush=True)

# =====================================================================
# FIGURE 3: CSI PHASE DENOISING (piw_phase_denoising.png)
# =====================================================================
def fig3_phase_denoising(meta):
    print("[RUN] Generating Figure 3: Multi-Antenna CSI Phase Denoising Performance...", flush=True)
    info = meta[0]
    try:
        csi_arr, _ = load_mat(info["csi_path"])
    except Exception as e:
        print(f"  [WARN] Failed to extract CSI matrix for Fig 3: {e}")
        return

    # subplots 1 hàng 2 cột
    fig, (ax_raw, ax_denoi) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.patch.set_facecolor("#FFFFFF") 
    
    ax_raw.set_facecolor("#FFFFFF")
    ax_denoi.set_facecolor("#FFFFFF")
    
    # Duyệt qua cả 3 Ăng-ten thu (RX 0, 1, 2) để trích xuất 3 đường tín hiệu
    for rx_idx in range(3):
        # Lấy vector 30 sóng mang con tại gói tin thời gian số 0
        csi_vector = csi_arr[0, 0, rx_idx, 0, :, 0]
        
        pha_raw, pha_denoised = linear_phase_clean(csi_vector)
        
        p_color = STYLE["paper_colors"][rx_idx]
        
        ax_raw.plot(np.arange(1, 31), pha_raw, color=p_color, linewidth=1.1, alpha=0.85, zorder=3)
        
        ax_denoi.plot(np.arange(1, 31), pha_denoised, color=p_color, linewidth=1.1, alpha=0.85, zorder=3)

    ax_raw.set_xlabel("subcarrier", fontsize=12, fontstyle="italic")
    ax_raw.set_ylabel("phase", fontsize=12, fontstyle="italic")
    ax_raw.set_xlim([0, 30])
    ax_raw.set_ylim([-3.2, 3.2])
    ax_raw.set_xticks([0, 10, 20, 30])
    ax_raw.set_yticks([-2, 0, 2])
    
    ax_raw.set_title("(a) raw csi phase", fontsize=12, y=-0.25, fontweight="bold")
    ax_raw.tick_params(direction='in', top=True, right=True)

    ax_denoi.set_xlabel("subcarrier", fontsize=12, fontstyle="italic")
    ax_denoi.set_ylabel("phase", fontsize=12, fontstyle="italic")
    ax_denoi.set_xlim([0, 30])
    ax_denoi.set_ylim([-3.2, 3.2])
    ax_denoi.set_xticks([0, 10, 20, 30])
    ax_denoi.set_yticks([-2, 0, 2])
    
    ax_denoi.set_title("(b) denoised csi phase", fontsize=12, y=-0.25, fontweight="bold")
    ax_denoi.tick_params(direction='in', top=True, right=True)

    fig.suptitle("CSI Phase Linear Transformation Denoising", fontsize=13, fontweight="bold", color=STYLE["text"], y=0.98)

    plt.tight_layout()
    
    out = os.path.join(OUT_DIR, "piw_phase_denoising.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    print(f"[SAVE] Figure 3: {out}", flush=True)
    
def main():
    print("=" * 75)
    print("  INITIALIZING CVPR PAPER COMPLIANT 3-FIGURE EDA VISUALIZATION PIPELINE  ")
    print("=" * 75)
    
    meta, mat_files, npy_files = scan_dataset()
    if not meta:
        print("[ERROR] Failed to compile paired index metadata mapping list. Aborting.")
        return
        
    fig1_overview(meta, mat_files, npy_files)
    fig2_keypoint_geometry(meta)
    fig3_phase_denoising(meta)
    
    print("\n" + "=" * 70)
    print("*** XỬ LÝ THÀNH CÔNG: Đã lưu ảnh vào /results_person_in_wifi.")
    print("=" * 70)

if __name__ == "__main__":
    main()