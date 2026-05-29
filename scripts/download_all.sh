#!/bin/bash
# Run on VPS: download all datasets to Bronze
# Usage: bash /opt/rfpose/download_all.sh

LOG=/opt/rfpose/download.log
echo "=== Download started: $(date) ===" | tee -a "$LOG"

# ── Wi-Pose ──
echo "[1/4] Wi-Pose (Google Drive, ~1.5 GB)" | tee -a "$LOG"
cd /opt/rfpose/data/bronze/public/wipose/raw_mat
rm -f *.part*
gdown 'https://drive.google.com/uc?id=1WL6bJ-rSVdsclRt9RFc0l5hhtXYmfNg9' -O wipose_data.zip 2>&1 | tee -a "$LOG"
if file wipose_data.zip | grep -q Zip; then
    unzip -q -o wipose_data.zip 2>&1 | tee -a "$LOG"
    rm -f wipose_data.zip
    echo "Wi-Pose OK: $(du -sh .)" | tee -a "$LOG"
else
    echo "Wi-Pose FAILED" | tee -a "$LOG"
fi

# ── MM-Fi ──
echo "" | tee -a "$LOG"
echo "[2/4] MM-Fi (Google Drive folder, ~80-120 GB)" | tee -a "$LOG"
cd /opt/rfpose/data/bronze/public/mmfi/raw
gdown --folder 'https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz' 2>&1 | tee -a "$LOG"
echo "MM-Fi: $(du -sh .)" | tee -a "$LOG"

# ── WiAR (already done via git) ──
echo "" | tee -a "$LOG"
echo "[3/4] WiAR - already cloned" | tee -a "$LOG"
echo "WiAR: $(du -sh /opt/rfpose/data/bronze/public/wiar/raw/)" | tee -a "$LOG"

# ── Widar 3.0 ──
echo "" | tee -a "$LOG"
echo "[4/4] Widar 3.0 (Tsinghua)" | tee -a "$LOG"
cd /opt/rfpose/data/bronze/public/widar3/raw
rm -f *.zip
# Try wget with proper headers
wget --no-check-certificate -c 'https://tns.thss.tsinghua.edu.cn/widar3.0/data/CSI_data_part1.zip' 2>&1 | tee -a "$LOG"
wget --no-check-certificate -c 'https://tns.thss.tsinghua.edu.cn/widar3.0/data/CSI_data_part2.zip' 2>&1 | tee -a "$LOG"
wget --no-check-certificate -c 'https://tns.thss.tsinghua.edu.cn/widar3.0/data/BVP_data.zip' 2>&1 | tee -a "$LOG"

for f in *.zip; do
    if [ -f "$f" ] && file "$f" | grep -q Zip; then
        echo "Extracting $f..." | tee -a "$LOG"
        unzip -q -o "$f" && rm -f "$f"
    else
        echo "Invalid/missing: $f" | tee -a "$LOG"
    fi
done
echo "Widar3: $(du -sh .)" | tee -a "$LOG"

# ── Summary ──
echo "" | tee -a "$LOG"
echo "=== SUMMARY ===" | tee -a "$LOG"
du -sh /opt/rfpose/data/bronze/public/*/ | tee -a "$LOG"
echo "" | tee -a "$LOG"
du -sh /opt/rfpose/data/bronze/ | tee -a "$LOG"
df -h / | tee -a "$LOG"
echo "=== Done: $(date) ===" | tee -a "$LOG"
