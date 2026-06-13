#!/bin/bash
#SBATCH --job-name=rfpose-humanlike-train-v2
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%A_%a.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
ML_DIR="$PROJECT_ROOT/ml"
GOLD_DIR="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/.tmp" "$PROJECT_ROOT/.cache" "$PROJECT_ROOT/.cache/matplotlib"
export TMPDIR="$PROJECT_ROOT/.tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export MLFLOW_TRACKING_URI="http://207.180.243.242:5000"
export MLFLOW_S3_ENDPOINT_URL="http://207.180.243.242:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin123"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    DATASET="mmfi"
    CONFIG_NAME="humanlike_mmfi17_v2"
    CKPT_DIR="$PROJECT_ROOT/checkpoints/humanlike-mmfi17-v2"
    RUN_NAME="humanlike-mmfi17-v2"
    ;;
  1)
    DATASET="wipose"
    CONFIG_NAME="humanlike_wipose18_v2"
    CKPT_DIR="$PROJECT_ROOT/checkpoints/humanlike-wipose18-v2"
    RUN_NAME="humanlike-wipose18-v2"
    ;;
  *)
    echo "Bad SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac

echo "JobID=$SLURM_JOB_ID array=${SLURM_ARRAY_TASK_ID:-0} dataset=$DATASET host=$(hostname) start=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

if [ ! -f "$GOLD_DIR/$DATASET/x.npy" ] || [ ! -f "$GOLD_DIR/$DATASET/y.npz" ]; then
  echo "Missing Gold v2 for $DATASET under $GOLD_DIR" >&2
  exit 2
fi

source "$VENV_DIR/bin/activate"
cd "$ML_DIR"
pip install -e . --quiet 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p "$CKPT_DIR"

python -m rfpose.training.train_vit2d \
  --config-name "$CONFIG_NAME" \
  data.gold_dir="$GOLD_DIR" \
  training.device=cuda \
  training.batch_size=32 \
  training.num_workers=0 \
  training.checkpoint_dir="$CKPT_DIR" \
  mlflow.run_name="$RUN_NAME" \
  hydra.run.dir="$CKPT_DIR"

echo "Train done dataset=$DATASET at $(date)"
