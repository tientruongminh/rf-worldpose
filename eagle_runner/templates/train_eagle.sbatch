#!/bin/bash
#SBATCH --job-name=rfpose-{{ job_id }}
#SBATCH --partition={{ partition }}
#SBATCH --gres=gpu:{{ gpu_type }}:{{ gpus }}
#SBATCH --cpus-per-task={{ cpus }}
#SBATCH --mem={{ mem }}
#SBATCH --time={{ time_limit }}
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -eo pipefail

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

source /etc/profile 2>/dev/null || true

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
CONDA_ROOT="/mnt/storage_6/project_data/pl0501-01/miniconda3"
ML_DIR="$PROJECT_ROOT/ml"
GOLD_DIR="$PROJECT_ROOT/data/gold/{{ dataset_version }}"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/{{ job_id }}"
TRAIN_MODULE="{{ train_module }}"
CONFIG_NAME="{{ config_name }}"

export PIP_CACHE_DIR="$PROJECT_ROOT/.pip-cache"
export TMPDIR="$PROJECT_ROOT/.tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
mkdir -p "$PROJECT_ROOT/logs" "$CHECKPOINT_DIR" "$PIP_CACHE_DIR" "$TMPDIR" \
         "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$ML_DIR"

# --- Python env via conda (project storage, no home quota issues) ---
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
  echo "Installing dependencies (first run takes ~5 min) ..."
  pip install --no-cache-dir --upgrade pip wheel
  pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
  pip install --no-cache-dir einops polars omegaconf boto3
fi

echo "Re-installing rfpose-ml (editable) to pick up code changes ..."
pip install --no-cache-dir -e . 2>&1 | tail -1

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi || true

export MLFLOW_TRACKING_URI="{{ mlflow_tracking_uri }}"
export MLFLOW_S3_ENDPOINT_URL="http://207.180.243.242:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin123"
export RUN_ID="{{ job_id }}"

echo "=== Testing MLflow connectivity ==="
if curl -sf --connect-timeout 5 "$MLFLOW_TRACKING_URI/health" >/dev/null 2>&1 || \
   curl -sf --connect-timeout 5 "$MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/search" \
     -H "Content-Type: application/json" -d '{"max_results":1}' >/dev/null 2>&1; then
  echo "MLflow reachable at $MLFLOW_TRACKING_URI"
else
  echo "WARNING: MLflow unreachable — metrics will not be logged"
fi

cd "$ML_DIR"
python -m "$TRAIN_MODULE" \
  --config-name "$CONFIG_NAME" \
  data.gold_dir="$GOLD_DIR" \
  training.device=cuda \
  training.epochs={{ epochs }} \
  training.batch_size={{ batch_size }} \
  training.dry_run={{ dry_run }} \
  training.checkpoint_dir="$CHECKPOINT_DIR" \
  mlflow.run_name="{{ job_id }}" \
  hydra.run.dir="$CHECKPOINT_DIR"

echo "Done at $(date)"
