"""
EDA — RF-WorldPose Bronze Datasets: MMFi_Dataset

Gồm 7 biểu đồ:
  1. mmfi_cross_modal_dynamics.png (Pose vs WiFi của hành động A06)
  2-7. mmfi_progression_{A02|A06|A12|A17|A20|A26}.png
  Chest expansion|Mark time|Squat|Waving hand (L)|Throwing (L)|Jumping up

Đường dẫn từ configs.py.
Result saved: /results_mmfi
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import configs  

DATA_DIR = configs.MMFI_DATA_DIR
OUTPUT_DIR = configs.MMFI_RESULTS_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ĐỊNH NGHĨA KẾT NỐI KHỚP CỦA MMFI (17 JOINTS)
MMFI_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),       # Chân phải: Pelvis -> R_Hip -> R_Knee -> R_Ankle
    (0, 4), (4, 5), (5, 6),       # Chân trái: Pelvis -> L_Hip -> L_Knee -> L_Ankle
    (0, 7), (7, 8),               # Cột sống: Pelvis -> Spine -> Thorax/Neck
    (8, 9), (9, 10),              # Đầu: Thorax/Neck -> Nose -> Head Top
    (8, 11), (11, 12), (12, 13),  # Tay trái: Thorax/Neck -> L_Shoulder -> L_Elbow -> L_Wrist
    (8, 14), (14, 15), (15, 16)   # Tay phải: Thorax/Neck -> R_Shoulder -> R_Elbow -> R_Wrist
]

# Từ điển ánh xạ tên hành động 
ACTION_MAPPING = {
    "A01": "Stretching", "A02": "Chest Exp (H)", "A03": "Chest Exp (V)",
    "A04": "Twist (L)", "A05": "Twist (R)", "A06": "Mark Time",
    "A07": "Limb Ext (L)", "A08": "Limb Ext (R)", "A09": "Lunge (LF)",
    "A10": "Lunge (RF)", "A11": "Limb Ext (Both)", "A12": "Squat",
    "A13": "Raising Hand (L)", "A14": "Raising Hand (R)",
    "A17": "Waving Hand (L)", "A19": "Picking Up", 
    "A20": "Throwing (L)", "A26": "Jumping Up"
}

def save(fig: plt.Figure, name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {path}")

def get_bone_color(u, v):
    """Phân bổ dải màu cơ thể chuẩn theo Human3.6M (Cam: Phải, Xanh dương: Trái, Xanh lá: Trục giữa)"""
    right_side = {1, 2, 3, 14, 15, 16}
    left_side = {4, 5, 6, 11, 12, 13}
    
    if u in right_side or v in right_side:
        return '#ff7f0e'  # Cam sáng (Bên phải cơ thể)
    elif u in left_side or v in left_side:
        return '#1f77b4'  # Xanh dương (Bên trái cơ thể)
    else:
        return '#2ca02c'  # Xanh lá (Trục xương sống giữa)

def main():
    print("=" * 70)
    print("ĐANG CHẠY EDA PHÂN TÍCH TIẾN TRÌNH THỜI GIAN THEO HÀNH ĐỘNG - MMFI")
    print("=" * 70)

    # Đề phòng trường hợp giải nén bị lồng thư mục MMFi_Dataset/MMFi_Dataset
    working_data_dir = DATA_DIR
    if not os.path.exists(os.path.join(working_data_dir, "E01")) and os.path.exists(os.path.join(working_data_dir, "MMFi_Dataset")):
        working_data_dir = os.path.join(working_data_dir, "MMFi_Dataset")

    if not os.path.isdir(working_data_dir):
        print(f"[ERROR]: Không tìm thấy thư mục MMFi tại: {working_data_dir}")
        return

    print(f"  [INFO] Thư mục Dữ liệu: {working_data_dir}")
    print(f"  [INFO] Thư mục Đầu ra: {OUTPUT_DIR}")

    target_actions = ['A02', 'A06', 'A12', 'A17', 'A20', 'A26']
    sample_dict = {}
    
    # Tìm kiếm ngẫu nhiên 1 file đại diện cho từng hành động
    for act in target_actions:
        search_pattern = os.path.join(working_data_dir, "E*", "S*", act, "ground_truth.npy")
        files = sorted(glob.glob(search_pattern))
        if files:
            sample_dict[act] = files[0]  # Lấy mẫu đầu tiên tìm thấy
        else:
            print(f"  [ERROR]: Không tìm thấy dữ liệu cho hành động {act}")

    if not sample_dict:
        print("[ERROR]: Không tìm thấy bất kỳ dữ liệu nào cho các hành động mục tiêu.")
        return

    # =====================================================================
    # FIGURE 1: POSE VS WIFI (mmfi_cross_modal_dynamics.png)
    # =====================================================================
    dyn_act = 'A06' if 'A06' in sample_dict else list(sample_dict.keys())[0]
    sample_file = sample_dict[dyn_act]
    sample_parts = sample_file.split(os.sep)
    sample_dir = os.path.dirname(sample_file)
    
    gt_sample = np.load(sample_file, allow_pickle=True)
    T_len = gt_sample.shape[0]
    
    csi_files = glob.glob(os.path.join(sample_dir, "wifi-csi", "*.npy"))
    csi_data = None
    if csi_files:
        try:
            csi_data = np.load(csi_files[0], allow_pickle=True)
        except Exception:
            pass

    fig1, ax1 = plt.subplots(figsize=(11, 5))
    
    pose_velocity = np.diff(gt_sample, axis=0)
    pose_energy = np.sum(np.linalg.norm(pose_velocity, axis=2), axis=1)
    time_axis = np.arange(1, T_len)
    
    color_pose = '#1f77b4'
    ax1.set_xlabel('Time Frame')
    ax1.set_ylabel('Body Kinetic Energy', color=color_pose, fontweight="bold")
    line1 = ax1.plot(time_axis, pose_energy, color=color_pose, linewidth=1.8, label="Pose Kinetic")
    ax1.tick_params(axis='y', labelcolor=color_pose)
    ax1.grid(True, alpha=0.2)
    
    ax2 = ax1.twinx()  
    color_csi = '#2ca02c'
    ax2.set_ylabel('WiFi Variance', color=color_csi, fontweight="bold")
    
    if csi_data is not None:
        csi_flat = csi_data.reshape(csi_data.shape[0], -1)
        if csi_flat.shape[0] != T_len:
            indices = np.linspace(0, csi_flat.shape[0] - 1, T_len - 1, dtype=int)
            csi_flat_synced = csi_flat[indices]
        else:
            csi_flat_synced = csi_flat[1:]
            
        csi_variance = np.std(csi_flat_synced, axis=1)
        line2 = ax2.plot(time_axis, csi_variance, color=color_csi, linewidth=1.2, linestyle="--", label="WiFi Signal")
        ax2.tick_params(axis='y', labelcolor=color_csi)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
    else:
        ax2.text(0.5, 0.5, "No WiFi Data Loaded", ha="center", va="center", color="gray")
        ax2.tick_params(axis='y', labelcolor='gray')

    act_name = ACTION_MAPPING.get(sample_parts[-2], sample_parts[-2])
    plt.title(f"FIGURE 1: Pose vs WiFi Dynamics\n[{sample_parts[-4]}] {sample_parts[-3]} - {act_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig1, "mmfi_cross_modal_dynamics")

    # =====================================================================
    # 7 FIGURE: Keypoints of 7 activities progression
    # =====================================================================
    for act, file_path in sample_dict.items():
        parts = file_path.split(os.sep)
        env, subject, action = parts[-4], parts[-3], parts[-2]
        action_name = ACTION_MAPPING.get(action, action)
        
        gt_data = np.load(file_path, allow_pickle=True)
        T_len = gt_data.shape[0]
        
        # Chọn 6 mốc thời gian rải đều từ đầu đến cuối hành động
        frame_indices = np.linspace(0, T_len - 1, 6, dtype=int)
        
        # GIỚI HẠN KHÔNG GIAN CỐ ĐỊNH cho toàn bộ sample
        X_all = gt_data[:, :, 0]
        Y_all = gt_data[:, :, 2]
        Z_all = -gt_data[:, :, 1]
        
        mid_x = (X_all.max() + X_all.min()) / 2.0
        mid_y = (Y_all.max() + Y_all.min()) / 2.0
        mid_z = (Z_all.max() + Z_all.min()) / 2.0
        max_range = np.array([X_all.max()-X_all.min(), Y_all.max()-Y_all.min(), Z_all.max()-Z_all.min()]).max() / 2.0

        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor('white')
        
        for sub_idx, f_idx in enumerate(frame_indices):
            pts = gt_data[f_idx]
            
            ax = fig.add_subplot(2, 3, sub_idx + 1, projection='3d')
            ax.set_facecolor('white')
            
            X_plot = pts[:, 0]
            Y_plot = pts[:, 2]
            Z_plot = -pts[:, 1]
            
            node_colors = plt.cm.tab20(np.linspace(0, 1, 17))
            ax.scatter(X_plot, Y_plot, Z_plot, c=node_colors, s=45, zorder=5, edgecolors='k', linewidths=0.5)
            
            for u, v in MMFI_SKELETON_CONNECTIONS:
                if u < len(pts) and v < len(pts):
                    b_color = get_bone_color(u, v)
                    ax.plot([X_plot[u], X_plot[v]], [Y_plot[u], Y_plot[v]], [Z_plot[u], Z_plot[v]], 
                            color=b_color, linewidth=2.5, alpha=0.8, zorder=3)
            
            progress_pct = int((f_idx / (T_len - 1)) * 100)
            ax.set_title(f"Frame {f_idx} (Progress: {progress_pct}%)", fontsize=11, fontweight="bold", color="#d62728")
            
            ax.set_xlabel("X (Width)", fontsize=8)
            ax.set_ylabel("Y (Depth)", fontsize=8)
            ax.set_zlabel("Z (Height)", fontsize=8)
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.view_init(elev=15, azim=-65)
            ax.grid(True, linestyle="--", alpha=0.2)

        fig.suptitle(f"MMFI Time Progression: {action_name}\n({env} - {subject})", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        save(fig, f"mmfi_progression_{act}")

    print("\n" + "=" * 70)
    print("*** XỬ LÝ THÀNH CÔNG: Đã xuất toàn bộ biểu đồ vào /results_mmfi.")
    print("=" * 70)

if __name__ == "__main__":
    main()