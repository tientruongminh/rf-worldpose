"""
EDA Runner
"""

import os
import sys
import time
import traceback

# ---------- Thiết lập môi trường và cấu hình hệ thống ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import configs

def run_module_main(module_name):
    print("\n" + "=" * 80)
    print(f"*** STARTING EDA MODULE: {module_name}")
    print("=" * 80)
    
    start_time = time.time()
    try:
        # Import động module (ví dụ: eda_mmfi)
        module = __import__(module_name)
        
        # Kiểm tra module có hàm main() không
        if hasattr(module, 'main'):
            module.main()
            elapsed_time = time.time() - start_time
            print(f"=> [SUCCESS] {module_name} executed successfully in {elapsed_time:.2f}s.")
            return True
        else:
            print(f"[WARNING] {module_name} does not have a main() function defined.")
            return False
            
    except ImportError as e:
        print(f"[ERROR] Could not import {module_name}. Make sure the file exists.")
        print(f"   Details: {e}")
        return False
    except Exception as e:
        print(f"[CRITICAL] Runtime error occurred while executing {module_name}:")
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("               RF-WORLDPOSE GENERAL EDA PIPELINE EXECUTOR               ")
    print("=" * 80)
    print(f"  [INFO] Base Datasets Directory (BRONZE): {configs.BRONZE_DIR}")
    print(f"  [INFO] Global Results Directory:         {configs.RESULTS_DIR}")
    
    # Danh sách các file EDA cần chạy
    eda_modules = [
        "eda_mmfi",
        "eda_wipose",
        "eda_wimans",
        "eda_wiar",
        "eda_uthar",
        "eda_person_in_wifi"
    ]
    
    summary = {}
    
    for module in eda_modules:
        success = run_module_main(module)
        summary[module] = "SUCCESS" if success else "FAILED"
        
    # --- In bảng tổng hợp kết quả chạy (Pipeline Summary) ---
    print("\n" + "=" * 50)
    print("              EDA PIPELINE SUMMARY            ")
    print("=" * 50)
    for module, status in summary.items():
        print(f"  MODULE: {module.ljust(20)} | STATUS: {status}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()