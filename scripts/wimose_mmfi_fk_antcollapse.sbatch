#!/bin/bash
#SBATCH --job-name=wimose-fk-antic
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
NFS_PROTO1="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2-proto1"
LOCAL_GOLD="/tmp/rfpose_gold_mmfi_fk_${SLURM_JOB_ID}"
CKPT_DIR="$PROJECT_ROOT/checkpoints/wimose-mmfi17j-fk-antcollapse-v1"

mkdir -p "$PROJECT_ROOT/logs" "$CKPT_DIR"
export MLFLOW_TRACKING_URI="http://207.180.243.242:5000"
export MLFLOW_S3_ENDPOINT_URL="http://207.180.243.242:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin123"

echo "=== WiMose FK + anti-collapse | Job $SLURM_JOB_ID | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/ml:${PYTHONPATH:-}"

python3 "$PROJECT_ROOT/scripts/assign_subject_splits.py" \
  --gold-root "$NFS_PROTO1" \
  --datasets mmfi \
  --protocol 1

rm -rf "$LOCAL_GOLD"
mkdir -p "$LOCAL_GOLD"
cp -a "$NFS_PROTO1/mmfi" "$LOCAL_GOLD/mmfi"
du -sh "$LOCAL_GOLD"

cd "$PROJECT_ROOT/ml/rfpose/training"

torchrun --standalone --nproc_per_node=4 train_wimose.py \
  --config-name wimose_mmfi17j_fk_antcollapse_eagle \
  data.gold_dir="$LOCAL_GOLD" \
  training.checkpoint_dir="$CKPT_DIR" \
  training.batch_size=48 \
  training.num_workers=2 \
  training.lr=2.4e-3

echo "=== Done $(date) ==="
