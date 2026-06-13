#!/bin/bash
#SBATCH --job-name=rfpose-humanlike-etl-v2
#SBATCH --partition=proxima-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
OUT_ROOT="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2"
WIPOSE_BRONZE="$PROJECT_ROOT/data/bronze/public/wipose/raw_mat/Wi-Pose-humanlike-src"
MMFI_SOURCE_GOLD="$PROJECT_ROOT/data/gold/rfpose-mmfi-17j-v1"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/.tmp" "$PROJECT_ROOT/.cache" "$OUT_ROOT"
export TMPDIR="$PROJECT_ROOT/.tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"

echo "JobID=$SLURM_JOB_ID host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-1} start=$(date)"
cd "$PROJECT_ROOT"

source "$VENV_DIR/bin/activate"
pip install -e ml/ --quiet 2>/dev/null || true
python - <<'PY' || pip install --no-cache-dir h5py
import h5py
PY
python - <<'PY'
import h5py, numpy
print('python deps ok', 'h5py', h5py.__version__, 'numpy', numpy.__version__)
PY

if [ ! -d "$WIPOSE_BRONZE" ]; then
  echo "Missing WiPose bronze at $WIPOSE_BRONZE" >&2
  exit 2
fi
if [ ! -d "$MMFI_SOURCE_GOLD/mmfi" ]; then
  echo "Missing MM-Fi source gold at $MMFI_SOURCE_GOLD/mmfi" >&2
  exit 2
fi

python scripts/build_humanlike_gold_v2.py \
  --dataset all \
  --wipose-bronze "$WIPOSE_BRONZE" \
  --mmfi-source-gold "$MMFI_SOURCE_GOLD" \
  --out-root "$OUT_ROOT" \
  --workers "${SLURM_CPUS_PER_TASK:-32}"

echo "ETL done at $(date)"
find "$OUT_ROOT" -maxdepth 2 -type f \( -name 'manifest.json' -o -name 'normalization.json' -o -name 'skeleton.json' \) -print -exec sed -n '1,80p' {} \;
