#!/bin/bash
#SBATCH --job-name=rfpose-eval-mmfi
#SBATCH --partition=proxima
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -eo pipefail

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
date

source /etc/profile 2>/dev/null || true

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
CONDA_ROOT="/mnt/storage_6/project_data/pl0501-01/miniconda3"
ML_DIR="$PROJECT_ROOT/ml"
GOLD_DIR="$PROJECT_ROOT/data/gold/rfpose-unified-v2"
CHECKPOINT="$PROJECT_ROOT/ml/checkpoints/best.pt"
OUTPUT_DIR="$PROJECT_ROOT/eval_results"

export PIP_CACHE_DIR="$PROJECT_ROOT/.pip-cache"
export TMPDIR="$PROJECT_ROOT/.tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
mkdir -p "$OUTPUT_DIR" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$ML_DIR"

# --- Python env ---
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_MARKER="$VENV_DIR/.conda-origin"

_need_fresh_venv() {
  [ ! -d "$VENV_DIR" ] && return 0
  [ ! -f "$VENV_MARKER" ] && return 0
  [ "$(cat "$VENV_MARKER" 2>/dev/null)" != "$CONDA_ROOT" ] && return 0
  return 1
}

if _need_fresh_venv; then
  echo "Creating fresh venv from $CONDA_ROOT/bin/python3 ..."
  rm -rf "$VENV_DIR"
  "$CONDA_ROOT/bin/python3" -m venv "$VENV_DIR"
  echo "$CONDA_ROOT" > "$VENV_DIR/.conda-origin"
fi
source "$VENV_DIR/bin/activate"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Installing dependencies ..."
  pip install --no-cache-dir --upgrade pip wheel
  pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
  pip install --no-cache-dir einops polars omegaconf boto3
fi

echo "Re-installing rfpose-ml (editable) ..."
pip install --no-cache-dir -e . 2>&1 | tail -1

# Install sklearn for action metrics
pip install --no-cache-dir scikit-learn 2>&1 | tail -1

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi || true

echo "=== Running MM-Fi Protocol Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Gold dir:   $GOLD_DIR"
echo ""

python -m rfpose.evaluation.eval_mmfi_protocols \
  --checkpoint "$CHECKPOINT" \
  --gold-dir "$GOLD_DIR" \
  --settings S1 S2 S3 \
  --protocols P3 \
  --batch-size 64 \
  --num-workers 4 \
  --output "$OUTPUT_DIR/eval_mmfi_$(date +%Y%m%d_%H%M%S).json"

echo "Done at $(date)"
