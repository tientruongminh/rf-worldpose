#!/bin/bash
#SBATCH --job-name=ngoc-transformer-proto1
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
NFS_PROTO1="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2-proto1"
LOCAL_GOLD="/tmp/rfpose_gold_mmfi_proto1_${SLURM_JOB_ID}"
CKPT_DIR="$PROJECT_ROOT/checkpoints/ngoc-transformer-mmfi17-proto1-v1"

mkdir -p "$PROJECT_ROOT/logs" "$CKPT_DIR"
export MLFLOW_TRACKING_URI="http://207.180.243.242:5000"
export MLFLOW_S3_ENDPOINT_URL="http://207.180.243.242:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin123"

echo "=== Ngọc Transformer Proto-1 2×GPU | Job $SLURM_JOB_ID | $(date) ==="
echo "Node: $(hostname) GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/ml:${PYTHONPATH:-}"

python3 "$PROJECT_ROOT/scripts/assign_subject_splits.py" \
  --gold-root "$NFS_PROTO1" --datasets mmfi --protocol 1

rm -rf "$LOCAL_GOLD" && mkdir -p "$LOCAL_GOLD"
cp -a "$NFS_PROTO1/mmfi" "$LOCAL_GOLD/mmfi"
du -sh "$LOCAL_GOLD"

cd "$PROJECT_ROOT/ml"
torchrun --standalone --nproc_per_node=2 -m rfpose.training.transformer_train \
  --config-name ngoc_transformer_eagle \
  data.gold_dir="$LOCAL_GOLD" \
  training.checkpoint_dir="$CKPT_DIR" \
  training.num_workers=2 \
  mlflow.run_name="ngoc-transformer-mmfi17-proto1-2gpu-v1"

echo "=== Done $(date) ==="
