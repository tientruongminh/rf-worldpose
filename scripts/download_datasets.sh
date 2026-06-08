#!/bin/bash
LOG=/opt/rfpose/download.log
exec > >(tee -a "$LOG") 2>&1
echo "=========================================="
echo "Download started: $(date)"
echo "=========================================="

cd /opt/rfpose/data/bronze/public

echo ""
echo "[1/4] Downloading Wi-Pose..."
if [ -f "wipose/.done" ]; then
    echo "  Wi-Pose already downloaded, skipping."
else
    cd wipose/raw_mat
    gdown "https://drive.google.com/uc?id=1WL6bJ-rSVdsclRt9RFc0l5hhtXYmfNg9" -O wipose_data.zip || {
        echo "  [WARN] gdown failed. Trying alternate method..."
        gdown "1WL6bJ-rSVdsclRt9RFc0l5hhtXYmfNg9" -O wipose_data.zip || {
            echo "  [FAIL] Wi-Pose needs manual download:"
            echo "  https://drive.google.com/file/d/1WL6bJ-rSVdsclRt9RFc0l5hhtXYmfNg9/view"
        }
    }
    if [ -f wipose_data.zip ]; then
        echo "  Extracting Wi-Pose..."
        unzip -q -o wipose_data.zip
        echo "  Wi-Pose extracted. Cleaning zip..."
        rm -f wipose_data.zip
    fi
    cd /opt/rfpose/data/bronze/public
    touch wipose/.done
    echo "  Wi-Pose step complete."
fi
echo "Disk: $(df -h / | tail -1)"
echo ""

echo "[2/4] Downloading MM-Fi..."
if [ -f "mmfi/.done" ]; then
    echo "  MM-Fi already downloaded, skipping."
else
    cd mmfi/raw
    gdown --folder "https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz" || {
        echo "  [WARN] MM-Fi folder download failed or partial."
        echo "  This dataset is very large (~80-120 GB)."
        echo "  If Google Drive quota exceeded, wait 24h or use Baidu:"
        echo "  https://pan.baidu.com/s/1IU9okQzdeCIaF7xCr1X_pw?pwd=t316"
    }
    cd /opt/rfpose/data/bronze/public
    touch mmfi/.done
    echo "  MM-Fi step complete."
fi
echo "Disk: $(df -h / | tail -1)"
echo ""

echo "[3/4] Downloading WiAR..."
if [ -f "wiar/.done" ]; then
    echo "  WiAR already downloaded, skipping."
else
    cd wiar/raw
    git clone --depth 1 https://github.com/linteresa/WiAR.git . 2>/dev/null || {
        git pull 2>/dev/null || true
    }
    cd /opt/rfpose/data/bronze/public
    touch wiar/.done
    echo "  WiAR complete."
fi
echo "Disk: $(df -h / | tail -1)"
echo ""

echo "[4/4] Downloading Widar 3.0..."
if [ -f "widar3/.done" ]; then
    echo "  Widar 3.0 already downloaded, skipping."
else
    cd widar3/raw
    echo "  Trying Tsinghua server..."
    wget -c "https://tns.thss.tsinghua.edu.cn/widar3.0/data/CSI_data_part1.zip" -O CSI_data_part1.zip 2>&1 || echo "  [WARN] Part 1 failed"
    wget -c "https://tns.thss.tsinghua.edu.cn/widar3.0/data/CSI_data_part2.zip" -O CSI_data_part2.zip 2>&1 || echo "  [WARN] Part 2 failed"
    wget -c "https://tns.thss.tsinghua.edu.cn/widar3.0/data/BVP_data.zip" -O BVP_data.zip 2>&1 || echo "  [WARN] BVP data failed"

    for f in *.zip; do
        if [ -f "$f" ] && [ "$f" != "*.zip" ]; then
            echo "  Extracting $f..."
            unzip -q -o "$f" && rm -f "$f" || echo "  [WARN] Extract failed: $f"
        fi
    done

    cd /opt/rfpose/data/bronze/public
    touch widar3/.done
    echo "  Widar 3.0 complete."
fi
echo "Disk: $(df -h / | tail -1)"
echo ""

echo "=========================================="
echo "DOWNLOAD SUMMARY - $(date)"
echo "=========================================="
echo ""
echo "Disk usage per dataset:"
du -sh /opt/rfpose/data/bronze/public/*/
echo ""
echo "Total Bronze:"
du -sh /opt/rfpose/data/bronze/
echo ""
df -h /
echo ""
echo "MANUAL DOWNLOAD NEEDED:"
echo "  Person-in-WiFi-3D: BaiduNetdisk only"
echo "  https://aiotgroup.github.io/Person-in-WiFi-3D/"
echo ""
echo "DONE."
