#!/usr/bin/env python3
"""
EDA — RF-WorldPose Bronze Datasets: WiMANS_Dataset

Gồm 4 figures (Overview, Co-occurrence, 2 CSI)

Đường dẫn từ configs.py.
Result saved: /results_wimans
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter

import configs

DATA_DIR = configs.WIMANS_DATA_DIR
OUT_DIR  = configs.WIMANS_RESULTS_DIR

os.makedirs(OUT_DIR, exist_ok=True)

def extract_pure_array(data):
    """ Hàm đệ quy để lấy mảng số liệu lõi từ cấu trúc MATLAB (Struct/Cell/Dict) """
    # 1. Xử lý trường hợp dữ liệu là Dictionary (do simplify_cells=True tạo ra từ Struct)
    if isinstance(data, dict):
        best_arr = None
        max_size = 0
        for k, v in data.items():
            if isinstance(k, str) and k.startswith('__'): 
                continue
            arr = extract_pure_array(v)
            if isinstance(arr, np.ndarray) and arr.size > max_size:
                max_size = arr.size
                best_arr = arr
        return best_arr if best_arr is not None else np.array([])
        
    # 2. Xử lý trường hợp dữ liệu là List
    if isinstance(data, list):
        return extract_pure_array(data[0]) if len(data) > 0 else np.array([])

    # 3. Xử lý trường hợp là mảng Numpy
    if isinstance(data, np.ndarray):
        if data.dtype.names is not None: 
            return extract_pure_array(data[data.dtype.names[0]])
        if data.dtype == 'O': 
            try:
                stacked = np.vstack(data)
                if stacked.dtype == 'O':
                    return extract_pure_array(stacked[0])
                return stacked
            except Exception:
                if data.size > 0:
                    return extract_pure_array(data.flat[0])
        return data
        
    return np.array(data)

def load_csi_file(file_path):
    """ Hàm đọc file CSI an toàn cho cả định dạng .npy và .mat """
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.mat'):
        import scipy.io
        try:
            mat_data = scipy.io.loadmat(file_path, simplify_cells=True)
        except TypeError:
            mat_data = scipy.io.loadmat(file_path)
            
        return extract_pure_array(mat_data)
    return None

def main():
    print("\n" + "=" * 70)
    print("WIMANS DATASET EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    data_path = os.path.join(DATA_DIR, "annotation.csv")
    
    print(f"[INFO] Đường dẫn dữ liệu nhãn: {data_path}")
    print(f"[INFO] Thư mục lưu kết quả ảnh: {OUT_DIR}")

    try:
        df = pd.read_csv(data_path)
        print(f"[INFO] Đọc thành công annotation.csv: {df.shape[0]} dòng x {df.shape[1]} cột")
    except FileNotFoundError:
        print(f"[INFO] Lỗi: Không tìm thấy file {data_path}. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Thiết lập style hiển thị cho biểu đồ
    sns.set_theme(style="whitegrid")
    act_cols = [col for col in df.columns if 'activity' in col and 'user' in col]

    # =====================================================================
    # FIGURE 1: OVERVIEW (wimans_overview.png)
    # =====================================================================
    print("[INFO] Đang tạo Figure 1: Overview...")
    fig1, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    sns.countplot(data=df, x='number_of_users', ax=axes[0], palette='viridis')
    axes[0].set_title('WIMANS - PHÂN BỐ SỐ LƯỢNG NGƯỜI DÙNG', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Số người', fontsize=10)
    axes[0].set_ylabel('Số lượng mẫu', fontsize=10)

    sns.countplot(data=df, x='environment', ax=axes[1], palette='Set2')
    axes[1].set_title('WIMANS - PHÂN BỐ THEO MÔI TRƯỜNG', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Môi trường', fontsize=10)
    axes[1].set_ylabel('Số lượng mẫu', fontsize=10)
    axes[1].tick_params(axis='x')

    wifi_counts = df['wifi_band'].value_counts()
    axes[2].pie(wifi_counts, labels=[f"{band} GHz" for band in wifi_counts.index], 
                autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90,
                textprops={'fontsize': 11})
    axes[2].set_title('WIMANS - TỶ LỆ SỬ DỤNG BĂNG TẦN WIFI', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig1_path = os.path.join(OUT_DIR, 'wimans_overview.png')
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  [SAVE] {fig1_path}")

    # =====================================================================
    # FIGURE 2: ACTIVITY DISTRIBUTION & CO-OCCURRENCE (wimans_activity_co_occurrence.png)
    # =====================================================================
    print("[INFO] Đang tạo Figure 2: Activity Distribution & Co-occurrence...")
    fig2, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 2.1 Tổng hợp phân bố hành động (gộp tất cả user)
    df_act = df[act_cols].melt(value_name='Activity').dropna()
    df_act = df_act[df_act['Activity'].str.strip() != '']
    
    sns.countplot(data=df_act, y='Activity', ax=axes[0], 
                  order=df_act['Activity'].value_counts().index, palette='magma')
    axes[0].set_title('WIMANS - PHÂN BỐ TỔNG HỢP HÀNH ĐỘNG', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tần suất xuất hiện', fontsize=10)
    axes[0].set_ylabel('Hành động', fontsize=10)

    # ma trận đồng xuất hiện hành động (Co-occurrence Matrix)
    co_occur_counter = Counter()
    for _, row in df.iterrows():
        acts = [str(x).strip() for x in row[act_cols].values if pd.notna(x) and str(x).strip() != '']
        if len(acts) > 1:
            for pair in combinations(sorted(acts), 2):
                co_occur_counter[pair] += 1

    unique_acts = sorted(list(set([a for pair in co_occur_counter.keys() for a in pair])))
    if unique_acts:
        co_df = pd.DataFrame(0, index=unique_acts, columns=unique_acts)
        for (a, b), count in co_occur_counter.items():
            co_df.loc[a, b] += count
            co_df.loc[b, a] += count
        
        sns.heatmap(co_df, annot=True, fmt='d', cmap='Reds', ax=axes[1], linewidths=.5)
        axes[1].set_title('WIMANS - MA TRẬN ĐỒNG XUẤT HIỆN HÀNH ĐỘNG', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Hành động A', fontsize=10)
        axes[1].set_ylabel('Hành động B', fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "Không đủ dữ liệu kịch bản nhiều người để lập ma trận", 
                     ha='center', va='center', fontsize=12)
        axes[1].set_title('WIMANS - MA TRẬN ĐỒNG XUẤT HIỆN HÀNH ĐỘNG', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig2_path = os.path.join(OUT_DIR, 'wimans_activity_co_occurrence.png')
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  [SAVE] {fig2_path}")

    # =====================================================================
    # FIGURE 3: CSI AMPLITUDE FROM AMP FOLDER (wimans_csi_amplitude.png)
    # =====================================================================
    print("[INFO] Đang cấu hình xử lý dữ liệu sóng CSI (Amp)...")
    amp_dir = os.path.join(DATA_DIR, "wifi_csi", "amp")
    amp_files = sorted(glob.glob(os.path.join(amp_dir, "*.npy"))) + sorted(glob.glob(os.path.join(amp_dir, "*.mat")))
    print(f"[INFO] Tìm thấy {len(amp_files)} file dữ liệu biên độ sóng trong thư mục 'amp'")

    fig3, axes3 = plt.subplots(3, 1, figsize=(15, 10))
    if amp_files:
        sample_amp = load_csi_file(amp_files[0])
        file_name = os.path.basename(amp_files[0])
        
        if sample_amp is not None and len(sample_amp.shape) == 4:
            for rx in range(min(3, sample_amp.shape[1])):
                for tx in range(min(3, sample_amp.shape[2])):
                    axes3[rx].plot(sample_amp[:, rx, tx, :], alpha=0.5, linewidth=0.7)
                axes3[rx].set_title(f"WIMANS - BIÊN ĐỘ CSI TẠI ĂNG-TEN NHẬN RX{rx}", fontsize=11, fontweight='bold')
                axes3[rx].set_xlabel("Time Samples", fontsize=9)
                axes3[rx].set_ylabel("Amplitude", fontsize=9)
        else:
            flat_data = sample_amp.reshape(sample_amp.shape[0], -1) if sample_amp is not None and sample_amp.size > 0 else np.zeros((100, 10))
            for i in range(min(10, flat_data.shape[1])):
                axes3[0].plot(flat_data[:, i], alpha=0.7)
            axes3[0].set_title("WIMANS - ĐỒ THỊ BIÊN ĐỘ CSI (DỮ LIỆU PHẲNG)")
            
        fig3.suptitle(f"WIMANS - BIÊN ĐỘ SÓNG CSI (AMPLITUDE) \n[File: {file_name}]", fontsize=14, fontweight='bold')
    else:
        for ax in axes3:
            ax.text(0.5, 0.5, "Không tìm thấy file mẫu trong thư mục wifi_csi/amp", ha='center', va='center')
        fig3.suptitle("WIMANS - BIÊN ĐỘ SÓNG CSI (AMPLITUDE)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig3_path = os.path.join(OUT_DIR, 'wimans_csi_amplitude.png')
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  [SAVE] {fig3_path}")

    # =====================================================================
    # FIGURE 4: CSI MATRIX FROM MAT FOLDER (wimans_csi_matrix.png)
    # =====================================================================
    print("[INFO] Đang cấu hình xử lý dữ liệu sóng CSI (Mat)...")
    mat_dir = os.path.join(DATA_DIR, "wifi_csi", "mat")
    mat_files = sorted(glob.glob(os.path.join(mat_dir, "*.npy"))) + sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
    print(f"[INFO] Tìm thấy {len(mat_files)} file ma trận sóng trong thư mục 'mat'")

    fig4, ax4 = plt.subplots(figsize=(12, 8))
    if mat_files:
        sample_mat = load_csi_file(mat_files[0])
        file_name_mat = os.path.basename(mat_files[0])
        
        if sample_mat is not None and sample_mat.size > 0:
            try:
                sample_mat = sample_mat.astype(complex)
            except Exception:
                pass

            if len(sample_mat.shape) == 4:
                display_data = sample_mat.mean(axis=(1, 2)).T
            elif len(sample_mat.shape) == 3:
                display_data = sample_mat.mean(axis=1).T
            else:
                display_data = sample_mat if len(sample_mat.shape) == 2 else sample_mat.reshape(sample_mat.shape[0], -1).T
            
            display_data_clean = np.abs(display_data).astype(float)
            
            im = ax4.imshow(display_data_clean, aspect="auto", cmap="plasma")
            fig4.colorbar(im, ax=ax4, label="CSI Value Intensity")
            ax4.set_title(f"WIMANS - MA TRẬN SÓNG CSI (MATRIX HEATMAP)\n[File: {file_name_mat}]", fontsize=13, fontweight='bold')
            ax4.set_xlabel("Time Samples", fontsize=10)
            ax4.set_ylabel("Subcarriers / Features Index", fontsize=10)
        else:
            ax4.text(0.5, 0.5, "Lỗi định dạng dữ liệu không thể bóc tách", ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, "Không tìm thấy file mẫu trong thư mục wifi_csi/mat", ha='center', va='center')
        ax4.set_title("WIMANS - MA TRẬN SÓNG CSI (MATRIX)", fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig4_path = os.path.join(OUT_DIR, 'wimans_csi_matrix.png')
    fig4.savefig(fig4_path, dpi=300)
    plt.close(fig4)
    print(f"  [SAVE] {fig4_path}")

    print("\n" + "=" * 70)
    print("*** XỬ LÝ THÀNH CÔNG: Đã lưu ảnh vào /results_wimans.")
    print("=" * 70)

if __name__ == '__main__':
    main()