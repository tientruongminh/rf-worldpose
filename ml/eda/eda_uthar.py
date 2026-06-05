"""
EDA — Human Activity Recognition - UT_HAR

Gồm 4 figures (Amplitude Heatmap, PCA Components, CSI Amplitude, Spectrogram STFT)

Đường dẫn từ configs.py.
Result saved: /results_uthar
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt

import configs

DATA_DIR = configs.UTHAR_DATA_DIR
OUTPUT_DIR = configs.UTHAR_RESULTS_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Thư mục Dữ liệu: {DATA_DIR}")
print(f"[INFO] Thư mục Đầu ra: {OUTPUT_DIR}")


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """1-D moving average via convolution. Window tự động clamp về len(data)."""
    window_size = min(int(window_size), len(data))
    window = np.ones(window_size) / float(window_size)
    return np.convolve(data, window, "same")


def load_sample(path: str) -> np.ndarray:
    """
    Đọc một file data trả về mảng amplitude shape (T, 90).
    """
    ext = os.path.splitext(path)[1].lower()

    def _is_npy_binary(p: str) -> bool:
        with open(p, "rb") as f:
            magic = f.read(6)
        return magic[:6] == b"\x93NUMPY"

    if ext == ".npy" or (ext == ".csv" and _is_npy_binary(path)):
        arr = np.load(path, allow_pickle=True)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[2])
        return arr.astype(float)
    elif ext == ".csv":
        df = pd.read_csv(path, header=None)
        amp = df.values[:, 1:91].astype(float)
        return amp
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {ext}")


def compute_pca_manual(amp: np.ndarray) -> np.ndarray:
    """PCA thủ công"""
    T, F = amp.shape
    constant_offset = np.empty_like(amp)
    for i in range(F):
        constant_offset[:, i] = moving_average(amp[:, i], 4000)

    filtered = amp - constant_offset
    for i in range(F):
        filtered[:, i] = moving_average(filtered[:, i], 10)

    cov_mat   = np.cov(filtered.T)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    idx       = eig_val.argsort()[::-1]
    eig_vec   = eig_vec[:, idx]

    pca_data  = filtered.dot(eig_vec)
    return pca_data

def visualize(path: str, save_prefix: str | None = None) -> None:
    """
    Visualize một file data UT_HAR: amplitude heatmap, PCA components,
    CSI amplitude, spectrogram (STFT).
    """
    amp = load_sample(path)
    T, F = amp.shape
    print(f"[INFO] Loaded: {os.path.basename(path)}  shape={amp.shape}  (full dataset)")

    if save_prefix is None:
        save_prefix = os.path.splitext(os.path.basename(path))[0]

    # =====================================================================
    # FIGURE 1: AMPLITUDE HEATMAP (uthar_amplitude.png)
    # =====================================================================
    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    slices = [(0, 29, "Antenna 1 Amplitude"),
              (30, 59, "Antenna 2 Amplitude"),
              (60, 89, "Antenna 3 Amplitude")]

    for ax, (s, e, title) in zip(axes, slices):
        im = ax.imshow(amp[:, s:e+1].T, interpolation="nearest",
                       aspect="auto", cmap="jet")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.suptitle("UT_HAR DATASET - AMPLITUDE HEATMAP", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = os.path.join(OUTPUT_DIR, f"uthar_amplitude_{save_prefix}.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"  -> Đã lưu: {out1}")

    # =====================================================================
    # FIGURE 2: PCA COMPONENTS (uthar_pca_components.png)
    # =====================================================================
    pca_data = compute_pca_manual(amp)

    xmin, xmax = 0, min(T, 20000)
    if xmax == xmin:
        xmax = T

    fig, axes = plt.subplots(6, 1, figsize=(18, 20))
    for idx, ax in enumerate(axes):
        ax.plot(pca_data[xmin:xmax, idx])
        ax.set_title(f"PCA {idx+1}{'st' if idx==0 else 'nd' if idx==1 else 'rd' if idx==2 else 'th'} component")
        ax.grid(True, linestyle="--", alpha=0.7)

    fig.suptitle("UT_HAR DATASET - PCA COMPONENTS", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = os.path.join(OUTPUT_DIR, f"uthar_pca_components_{save_prefix}.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"  -> Đã lưu: {out2}")
    

    # =====================================================================
    # FIGURE 3: CSI AMPLITUDE (uthar_csi_amplitude.png)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(amp[:, :10])
    ax.set_title("UT_HAR DATASET - CSI AMPLITUDE", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time-steps")
    ax.set_ylabel("Amplitude")
    
    fig.tight_layout()
    out3 = os.path.join(OUTPUT_DIR, f"uthar_csi_amplitude_{save_prefix}.png")
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"  -> Đã lưu: {out3}")

    # =====================================================================
    # FIGURE 4: SPECTROGRAM STFT (uthar_stft.png)
    # =====================================================================
    fig, axes = plt.subplots(6, 1, figsize=(18, 30))
    for idx, ax in enumerate(axes):
        plt.sca(ax)
        Pxx, freqs, bins, im = ax.specgram(
            pca_data[:, idx], NFFT=128, Fs=1000, noverlap=1,
            cmap="jet", vmin=-100, vmax=20
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(f"Spectrogram (STFT) — PCA component {idx+1}")
        fig.colorbar(im, ax=ax)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 100)

    fig.suptitle("UT_HAR DATASET - SPECTROGRAM (STFT)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out4 = os.path.join(OUTPUT_DIR, f"uthar_stft_{save_prefix}.png")
    fig.savefig(out4, dpi=150)
    plt.close(fig)
    print(f"  -> Đã lưu: {out4}")


def run_all() -> None:
    """Chỉ visualize file X_train.csv trong DATA_DIR."""
    path = os.path.join(DATA_DIR, "X_train.csv")

    if not os.path.isfile(path):
        print(f"[WARN] Không tìm thấy file: {path}")
        print("       Kiểm tra lại BRONZE_DIR trong configs.py.")
        return

    print(f"[INFO] Bắt đầu visualize: {path}\n")
    try:
        visualize(path)
    except Exception as e:
        print(f"[ERROR] X_train.csv: {e}")

    print(f"\n[DONE] Ảnh được lưu tại: {OUTPUT_DIR}")


def run_single(filename: str) -> None:
    """
    Visualize một file cụ thể (chỉ cần tên file, không cần full path).
    Ví dụ: run_single("X_test.npy")
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    visualize(path)


def main(file_name: str | None = None) -> None:
    """
    Hàm thực thi chính được gọi từ ngoài (ví dụ: main.py).
    """
    if file_name:
        run_single(file_name)
    else:
        run_all()
        
    print("\n" + "=" * 70)
    print("*** XỬ LÝ THÀNH CÔNG: Đã lưu ảnh vào /results_uthar.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PCA + STFT Visualization cho UT_HAR dataset"
    )
    parser.add_argument(
        "--file", "-f",
        default=None,
        help="Tên file cụ thể trong DATA_DIR (vd: X_test.npy). "
             "Nếu không truyền, xử lý toàn bộ file trong thư mục."
    )
    args = parser.parse_args()
    main(args.file)