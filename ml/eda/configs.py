"""
EDA configuration — paths & settings.
"""

import os

# ── 1. Thư mục gốc ──
BRONZE_DIR = "/home/huong/rf-worldpose/data/bronze"
RESULTS_DIR = "/home/huong/workspace/rfpose-shared/rf-worldpose/ml/eda/results_eda"

# ── 2. Đường dẫn chi tiết cho từng Dataset ──
# nối BRONZE_DIR với tên thư mục tương ứng của dataset

# WiAR Dataset
WIAR_DATA_DIR = os.path.join(BRONZE_DIR, "WiAR-master/WiAR-master/data/data")
WIAR_RESULTS_DIR = os.path.join(RESULTS_DIR, "results_wiar")

# MMFi Dataset
MMFI_DATA_DIR = os.path.join(BRONZE_DIR, "MMFi_Dataset") 
MMFI_RESULTS_DIR = os.path.join(RESULTS_DIR, "results_mmfi")

# MMFi Dataset
PIW_DATA_DIR = os.path.join(BRONZE_DIR, "wifipose_dataset") 
PIW_RESULTS_DIR = os.path.join(RESULTS_DIR, "results_person_in_wifi")

# WiMANS Dataset
WIMANS_DATA_DIR = os.path.join(BRONZE_DIR, "WiMANS")
WIMANS_RESULTS_DIR = os.path.join(RESULTS_DIR, "results_wimans")

# Wi-Pose Dataset
WIPOSE_DATA_DIR = os.path.join(BRONZE_DIR, "Wi-Pose", "Wi-Pose", "Train")
WIPOSE_RESULTS_DIR = os.path.join(RESULTS_DIR, "results_wipose")

# UT-HAR Dataset
UTHAR_DATA_DIR = os.path.join(BRONZE_DIR, "UT_HAR", "data")
UTHAR_RESULTS_DIR = os.path.join(RESULTS_DIR, "results_uthar")