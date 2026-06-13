#!/usr/bin/env python3
"""
EDA — RF-WorldPose Bronze Datasets: WiAR Dataset

Gồm 4 figures (Chuỗi thời gian biên độ CSI, Phổ tần số Spectrogram, Biểu đồ phân phối biên độ, Biểu đồ nhiệt sóng mang con)

Đường dẫn từ configs.py.
Result saved: /results_wiar
"""

import os
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft

import configs 

DATA_DIR = configs.WIAR_DATA_DIR
OUTPUT_DIR = configs.WIAR_RESULTS_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

PREFIX = "wiar"
SAMPLE_RATE = 30  

# 16 action của WiAR
ACTIONS = {
    1: "horizontal_arm_wave", 2: "high_arm_wave", 3: "two_hands_wave",
    4: "high_throw", 5: "draw_x", 6: "draw_tick",
    7: "toss_paper", 8: "forward_kick", 9: "side_kick",
    10: "bend", 11: "hand_clap", 12: "walk",
    13: "phone_call", 14: "drink_water", 15: "sit_down",
    16: "squat"
}

COMPARE_ACTIONS = [15, 16, 10]  # take 3 activities: sit_down, squat, bend
N_PACKETS_SHOW = 250


def parse_csi_file(filepath):
    """
    Đọc file .dat CSI của WiAR -> trả về mảng biên độ (dB) có kích thước (n_packets, 3, 30).

    Định dạng: mỗi bản ghi gồm 212 byte = 32 byte tiêu đề + 180 byte dữ liệu CSI.
    Dữ liệu CSI: 30 sóng mang con x 3 ăng-ten x giá trị biên độ uint16 (xen kẽ: ant0_sc0, ant0_sc1, ..., ant0_sc29, ant1_sc0, ...).
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    records = []
    offset = 0
    while offset < len(data) - 3:
        field_len = struct.unpack('>H', data[offset:offset + 2])[0]
        code = data[offset + 2]
        offset += 3

        if code == 187:  # bản ghi phản hồi beamforming
            rec = data[offset:offset + field_len - 1]
            if len(rec) < 180 + 32:
                offset += field_len - 1
                continue

            # Dữ liệu CSI bắt đầu từ byte thứ 32
            csi_bytes = rec[32:]

            # Trích xuất giá trị biên độ: 30 sóng mang con x 3 ăng-ten, kiểu uint16
            amp = np.zeros((30, 3), dtype=np.float64)
            for i in range(30):
                for a in range(3):
                    idx = (a * 30 + i) * 2
                    if idx + 2 <= len(csi_bytes):
                        amp[i, a] = struct.unpack('<H', csi_bytes[idx:idx + 2])[0]

            # Chuyển đổi sang dB: 20*log10(biên độ)
            amp_db = 20 * np.log10(amp + 1)

            records.append(amp_db)

        offset += field_len - 1

    if not records:
        return None

    stack = np.array(records)  # (n_packets, 30, 3)
    # Đổi thứ tự chiều dữ liệu thành (n_packets, 3, 30)
    stack = stack.transpose(0, 2, 1)

    # Xử lý ngoại lệ giống mã MATLAB: giá trị <-20 hoặc >70 -> gán bằng 25
    stack = np.where((stack < -20) | (stack > 70), 25, stack)
    return stack


def load_action(action_id, n_exec=30):
    results = []
    for vdir in sorted(os.listdir(DATA_DIR)):
        vpath = os.path.join(DATA_DIR, vdir, vdir)
        if not os.path.isdir(vpath):
            continue
        for e in range(1, n_exec + 1):
            fp = os.path.join(vpath, f"csi_a{action_id}_{e}.dat")
            if os.path.exists(fp):
                amp = parse_csi_file(fp)
                if amp is not None:
                    results.append(amp)
    return results


def load_multiple_actions(action_ids, n_exec=30):
    return {aid: load_action(aid, n_exec) for aid in action_ids}


def plot_amplitude_series(data, action_ids, savepath):
    """3 biểu đồ phụ: Biên độ CSI (dB) theo packet cho từng action."""
    fig, axes = plt.subplots(1, len(action_ids), figsize=(11, 3), dpi=150)
    for idx, aid in enumerate(action_ids):
        amps = data[aid]
        if not amps:
            continue
        combined = np.concatenate([a[:N_PACKETS_SHOW, 0, 15] for a in amps[:5]], axis=0)
        combined = combined[:N_PACKETS_SHOW]
        w = 5
        smoothed = np.convolve(combined, np.ones(w) / w, mode='same')
        ax = axes[idx]
        ax.plot(smoothed, 'k-', lw=0.8)
        
        ax.set_title(f"WiAR - {ACTIONS.get(aid, f'Action {aid}')}", fontsize=10)
        ax.set_xlabel("Số thứ tự Packet", fontsize=8)
        ax.set_ylabel("Biên độ CSI (dB)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, N_PACKETS_SHOW)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVE] {savepath}")


def plot_spectrograms(data, action_ids, savepath):
    """3 biểu đồ phụ: Phổ tần số (thời gian - tần số) cho từng action."""
    fig, axes = plt.subplots(1, len(action_ids), figsize=(11, 3), dpi=150)
    for idx, aid in enumerate(action_ids):
        amps = data[aid]
        if not amps:
            continue
        combined = np.concatenate([a[:N_PACKETS_SHOW, 0, 15] for a in amps[:5]], axis=0)
        combined = combined[:N_PACKETS_SHOW]
        centered = combined - np.mean(combined)
        f, t, Zxx = stft(centered, fs=SAMPLE_RATE, nperseg=30, noverlap=15)
        Zxx_db = 10 * np.log10(np.abs(Zxx) + 1e-10)
        ax = axes[idx]
        im = ax.pcolormesh(t, f, Zxx_db, cmap='viridis', shading='auto')

        ax.set_title(f"WiAR - Phổ tần số ({ACTIONS.get(aid, f'Action {aid}')})\nĐộ phân giải f = {f[1]-f[0]:.1f} Hz, t = {(t[-1]-t[0]):.3f} s", fontsize=8)
        ax.set_xlabel("Thời gian (s)", fontsize=8)
        ax.set_ylabel("Tần số (Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, max(f[:15]))
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVE] {savepath}")


def plot_amplitude_hist(data, action_ids, savepath):
    """Biểu đồ phân phối dạng histogram chồng lấp của biên độ CSI theo action."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, len(action_ids)))
    for i, aid in enumerate(action_ids):
        if aid not in data or not data[aid]:
            continue
        vals = np.concatenate([a[:200, 0, 15].flatten() for a in data[aid][:10]])
        ax.hist(vals, bins=60, alpha=0.45, label=ACTIONS.get(aid, str(aid)),
                color=colors[i], density=True)
    
    ax.set_title("WiAR - Phân phối Biên độ CSI", fontsize=10)
    ax.set_xlabel("Biên độ CSI (dB)", fontsize=9)
    ax.set_ylabel("Mật độ", fontsize=9)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVE] {savepath}")


def plot_subcarrier_heatmap(data, action_id, savepath):
    """Biểu đồ nhiệt (Heatmap): biên độ CSI trung bình của từng sóng mang con qua các packet."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    if action_id not in data or not data[action_id]:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return
    combined = np.mean([a[:N_PACKETS_SHOW, 0, :] for a in data[action_id][:10]], axis=0)
    im = ax.imshow(combined.T, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel("Chỉ số Packet", fontsize=9)
    ax.set_ylabel("Chỉ số Sóng mang con", fontsize=9)

    ax.set_title(f"WiAR - Biên độ sóng mang con ({ACTIONS.get(action_id, action_id)})", fontsize=10)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Biên độ (dB)", fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVE] {savepath}")


def main():
    print("=" * 60)
    print("Khám phá dữ liệu (EDA) - Tập dữ liệu WiAR")
    print("=" * 60)

    print("\n[INFO] Đã tải cấu hình và đường dẫn xử lý...")
    print(f"[INFO] Thư mục Dữ liệu: {DATA_DIR}")
    print(f"[INFO] Thư mục Đầu ra: {OUTPUT_DIR}")

    print("\n[INFO] Đang tải dữ liệu cho tất cả 16 action...")
    try:
        data = load_multiple_actions(list(range(1, 17)), n_exec=30)
    except FileNotFoundError:
        print(f"[ERROR] Không tìm thấy DATA_DIR tại {DATA_DIR}")
        return

    for aid in sorted(data):
        print(f"  [INFO] Action {aid:2d} ({ACTIONS[aid]:>20s}): {len(data[aid])} tệp")

    total = sum(len(v) for v in data.values())
    print(f"\n[INFO] Tổng số tệp đã tải: {total}")

    if total == 0:
        print("[ERROR] Không tìm thấy dữ liệu. Hãy kiểm tra lại DATA_DIR.")
        return

    print("\n[INFO] Đang tạo các biểu đồ...")

    # =====================================================================
    # FIGURE 1: CSI AMPLITUDE TIME-SERIES (wiar_csi_amplitude_timeseries.png)
    # =====================================================================
    plot_amplitude_series(
        data, COMPARE_ACTIONS,
        os.path.join(OUTPUT_DIR, f"{PREFIX}_csi_amplitude_timeseries.png")
    )

    # =====================================================================
    # FIGURE 2: SPECTROGRAMS (wiar_spectrogram.png)
    # =====================================================================
    plot_spectrograms(
        data, COMPARE_ACTIONS,
        os.path.join(OUTPUT_DIR, f"{PREFIX}_spectrogram.png")
    )

    # =====================================================================
    # FIGURE 3: AMPLITUDE DISTRIBUTION HISTOGRAM (wiar_amplitude_distribution.png)
    # =====================================================================
    plot_amplitude_hist(
        data, list(ACTIONS.keys()),
        os.path.join(OUTPUT_DIR, f"{PREFIX}_amplitude_distribution.png")
    )

    # =====================================================================
    # FIGURE 4: SUBCARRIER HEATMAP (wiar_subcarrier_heatmap.png)
    # =====================================================================
    plot_subcarrier_heatmap(
        data, COMPARE_ACTIONS[0],
        os.path.join(OUTPUT_DIR, f"{PREFIX}_subcarrier_heatmap.png")
    )

    print("\n" + "=" * 70)
    print("*** XỬ LÝ THÀNH CÔNG: Đã lưu ảnh vào /results_wiar.")
    print("=" * 70)

if __name__ == "__main__":
    main()