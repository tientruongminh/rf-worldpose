#!/bin/bash
#SBATCH --job-name=rfpose-mmfi-clean-etl
#SBATCH --partition=proxima-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
OUT_ROOT="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2"
MMFI_SOURCE_GOLD="$PROJECT_ROOT/data/gold/rfpose-mmfi-17j-v1"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/.tmp" "$PROJECT_ROOT/.cache" "$OUT_ROOT"
export TMPDIR="$PROJECT_ROOT/.tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"

echo "JobID=$SLURM_JOB_ID host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-1} start=$(date)"
cd "$PROJECT_ROOT"

source "$VENV_DIR/bin/activate"
pip install -e ml/ --quiet 2>/dev/null || true

python scripts/build_humanlike_gold_v2.py \
  --dataset mmfi \
  --mmfi-source-gold "$MMFI_SOURCE_GOLD" \
  --out-root "$OUT_ROOT" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "ETL done at $(date)"
python3 -m json.tool "$OUT_ROOT/mmfi/manifest.json" | sed -n '1,80p'
