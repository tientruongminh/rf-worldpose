"""
EDA — Wi-Pose Dataset

Gồm 5 figures (1 biểu đồ phân bố hành động, 3 biểu đồ Skeleton 3D Sequence cho bend/crouch/jump, 1 bản đồ nhiệt ma trận CSI)

Đường dẫn từ configs.py.
Result saved: /results_wipose
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

import configs 

DATA_DIR = configs.WIPOSE_DATA_DIR
OUT_DIR  = configs.WIPOSE_RESULTS_DIR

os.makedirs(OUT_DIR, exist_ok=True)

def load_wipose_sample(filepath):
    """Đọc file .mat định dạng HDF5 của tập Wi-Pose."""
    with h5py.File(filepath, 'r') as f:
        # Xử lý số phức của CSI
        csi_raw = f['CSI'][:]
        if csi_raw.dtype.names and 'real' in csi_raw.dtype.names:
            csi_complex = csi_raw['real'] + 1j * csi_raw['imag']
        else:
            csi_complex = np.array(f['CSI']) 
            
        # Xử lý Skeleton
        skeleton = np.array(f['SkeletonPoints'])
        skeleton = np.squeeze(skeleton)  # Loại bỏ chiều dư thừa
        
        # Xử lý chiều mảng về ma trận chuẩn (3, 18) -> [X, Y, Z]
        if skeleton.shape == (18, 3):
            skeleton = skeleton.T
        elif len(skeleton.shape) == 1 and skeleton.shape[0] == 54:
            skeleton = skeleton.reshape(3, 18)
            
    return csi_complex, skeleton


def get_wipose_bone_color(u, v):
    """Phân bổ dải màu cơ thể đối xứng cho 18 khớp Wi-Pose"""
    right_side = {2, 3, 4, 5, 6, 7, 14, 16}
    left_side = {8, 9, 10, 11, 12, 13, 15, 17}
    
    if u in right_side or v in right_side:
        return '#ff7f0e'  
    elif u in left_side or v in left_side:
        return '#1f77b4' 
    else:
        return '#2ca02c' 


def plot_6_skeletons_3d(skeletons_list, save_path, action_name=""):
    """Vẽ 6 bộ xương (Skeleton) trong cùng 1 figure (2 hàng x 3 cột)."""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')

    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4), 
        (1, 5), (5, 6), (6, 7), 
        (1, 8), (8, 9), (9, 10), 
        (1, 11), (11, 12), (12, 13), 
        (8, 11), (0, 14), (14, 16), (0, 15), (15, 17)
    ]
    node_colors = plt.cm.tab20(np.linspace(0, 1, 18))

    num_plots = min(6, len(skeletons_list))

    for i in range(num_plots):
        skeleton = skeletons_list[i]
        
        X_pixels = skeleton[0, :]
        Y_pixels = skeleton[1, :]
        C_scores = skeleton[2, :]

        # Map sang hệ trục 3D (Đứng thẳng)
        X_3d = Y_pixels
        Y_3d = np.zeros_like(X_pixels)
        Z_3d = -X_pixels

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.set_facecolor('white')
        
        # Nối xương
        for u, v in bones:
            if u < skeleton.shape[1] and v < skeleton.shape[1]:
                if C_scores[u] > 0.1 and C_scores[v] > 0.1:
                    b_color = get_wipose_bone_color(u, v)
                    ax.plot([X_3d[u], X_3d[v]], [Y_3d[u], Y_3d[v]], [Z_3d[u], Z_3d[v]], 
                            color=b_color, linewidth=3, alpha=0.8, zorder=3)

        ax.scatter(X_3d, Y_3d, Z_3d, c=node_colors, marker='o', s=C_scores*60, zorder=5, edgecolors='k')

        max_range = np.array([X_3d.max()-X_3d.min(), Z_3d.max()-Z_3d.min()]).max() / 2.0
        mid_x = (X_3d.max() + X_3d.min()) / 2.0
        mid_z = (Z_3d.max() + Z_3d.min()) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(-max_range, max_range)  
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_box_aspect([1, 1, 1])

        ax.set_title(f"Frame {i+1}", fontsize=11, fontweight="bold")
        ax.set_xlabel('X (Width)', fontsize=9)
        ax.set_ylabel('Virtual Depth', fontsize=9)
        ax.set_zlabel('Z (Height)', fontsize=9)
        
        # Góc nhìn
        ax.view_init(elev=10, azim=-60)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.yaxis.set_ticklabels([])

    title_str = f"Wi-Pose Skeleton Sequence - {action_name.capitalize()}" if action_name else "Wi-Pose Skeleton Samples"
    plt.suptitle(title_str, fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92) 
    
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_dataset_distribution(action_counts, save_path):
    """Vẽ Phân bố số lượng dữ liệu dựa trên file thực tế."""
    actions = sorted(action_counts.keys())
    amounts = [action_counts[a] for a in actions]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(actions)))

    plt.figure(figsize=(11, 5.5))
    plt.bar(actions, amounts, color=colors, width=0.55, edgecolor='gray', linewidth=0.5)
    
    max_amount = max(amounts) if amounts else 15000
    plt.ylim(0, max_amount * 1.1) 
    
    plt.ylabel('Amount / pcs', fontweight="bold")
    plt.xlabel('Action Categories', fontweight="bold")
    plt.title('Wi-Pose Dataset - Activity Distribution', fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_csi_matrix_heatmap(csi, save_path):
    """Vẽ bản đồ nhiệt (Heatmap) thể hiện biên độ ma trận CSI."""
    tx, rx = 0, 0
    amplitude_matrix = np.abs(csi[tx, rx, :, :])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(amplitude_matrix, cmap="viridis", cbar_kws={'label': 'Amplitude (dB)'})
    
    plt.title(f"CSI Amplitude Matrix (Tx={tx}, Rx={rx})", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Time Packet Index", fontweight="bold")
    plt.ylabel("Subcarrier Index (0-29)", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("RUNNING PIPELINE EDA FOR Wi-POSE DATASET")
    print("=" * 70)
    print(f"  [INFO] Đường dẫn nạp dữ liệu: {DATA_DIR}")
    print(f"  [INFO] Thư mục xuất kết quả ảnh: {OUT_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"  [ERROR] Không tìm thấy thư mục dữ liệu Wi-Pose tại: {DATA_DIR}")
        return

    mat_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mat')])
    
    if len(mat_files) == 0:
        print(f"  [WARNING] Thư mục trống, không tìm thấy file dữ liệu .mat nào để phân tích.")
        return
        
    print(f"  [INFO] Đang quét {len(mat_files)} file để thống kê phân bố hành động...")
    action_counts = Counter()
    for filename in mat_files:
        action_name = filename.split('_')[0]
        action_counts[action_name] += 1
        
    print(f"        -> Thống kê thực tế: {dict(action_counts)}")

    # =====================================================================
    # FIGURE 3: DATASET DISTRIBUTION (wipose_dataset_distribution.png)
    # =====================================================================
    dist_path = os.path.join(OUT_DIR, "wipose_dataset_distribution.png")
    plot_dataset_distribution(action_counts, dist_path)
    print(f"  [SAVE] {dist_path}")
    
    # =====================================================================
    # FIGURE 6: SKELETON 3D SEQUENCES (wipose_6_skeletons_3d_{action}.png)
    # =====================================================================
    target_actions = ['bend', 'crouch', 'jump']
    step = 50
    
    print(f"  [INFO] Bắt đầu trích xuất và vẽ Skeleton cho các hành động: {target_actions} (step={step})")
    
    for action in target_actions:
        action_files = [f for f in mat_files if f.startswith(f"{action}_")]
        
        if not action_files:
            print(f"  [WARNING] Bỏ qua {action}: Không tìm thấy file nào có tiền tố {action}_")
            continue
            
        skeletons_list = []
        for i in range(6):
            idx = i * step
            if idx >= len(action_files):
                idx = len(action_files) - 1
                
            sample_path = os.path.join(DATA_DIR, action_files[idx])
            csi, skeleton = load_wipose_sample(sample_path)
            skeletons_list.append(skeleton)
            
        skel_plot_path = os.path.join(OUT_DIR, f"wipose_6_skeletons_3d_{action}.png")
        plot_6_skeletons_3d(skeletons_list, skel_plot_path, action_name=action)
        print(f"  [SAVE] {skel_plot_path}")

    # =====================================================================
    # FIGURE 4: CSI AMPLITUDE MATRIX (wipose_csi_matrix.png)
    # =====================================================================
    print(f"  [INFO] Đang trích xuất ma trận CSI từ file mẫu...")
    sample_csi_path = os.path.join(DATA_DIR, mat_files[0])
    csi_data, _ = load_wipose_sample(sample_csi_path)
    
    heatmap_path = os.path.join(OUT_DIR, "wipose_csi_matrix.png")
    plot_csi_matrix_heatmap(csi_data, heatmap_path)
    print(f"  [SAVE] {heatmap_path}")

    print("\n" + "=" * 70)
    print("*** XỬ LÝ THÀNH CÔNG: Đã lưu ảnh vào /results_wipose.")
    print("=" * 70)

if __name__ == "__main__":
    main()