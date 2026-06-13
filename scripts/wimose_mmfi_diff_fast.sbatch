#!/bin/bash
#SBATCH --job-name=wimose-diff
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00      # 200 epochs diffusion training ≈ 1.5-2 days
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
NFS_PROTO1="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2-proto1"
LOCAL_GOLD="/tmp/rfpose_gold_mmfi_diff_${SLURM_JOB_ID}"
CKPT_DIR="$PROJECT_ROOT/checkpoints/wimose-mmfi17j-diff-v1"
MAE_ENCODER="$PROJECT_ROOT/checkpoints/wimae-mmfi-pretrain-v1/encoder_best.pt"

mkdir -p "$PROJECT_ROOT/logs" "$CKPT_DIR"
export MLFLOW_TRACKING_URI="http://207.180.243.242:5000"
export MLFLOW_S3_ENDPOINT_URL="http://207.180.243.242:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin123"

echo "=== WiMose Diffusion Decoder MM-Fi | Job $SLURM_JOB_ID | $(date) ==="
echo "Node: $(hostname) GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Check if pretrained encoder is available
if [ -f "$MAE_ENCODER" ]; then
    echo ">>> Pretrained MAE encoder found: $MAE_ENCODER"
    ENCODER_ARG="training.pretrained_encoder=$MAE_ENCODER"
else
    echo ">>> WARNING: No pretrained encoder at $MAE_ENCODER"
    echo ">>> Training encoder from scratch (consider running wimose_mmfi_mae_pretrain.sbatch first)"
    ENCODER_ARG="training.pretrained_encoder="
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/ml:${PYTHONPATH:-}"

# Apply Protocol-1 split (idempotent)
echo ">>> Applying Protocol-1 split..."
python3 "$PROJECT_ROOT/scripts/assign_subject_splits.py" \
  --gold-root "$NFS_PROTO1" \
  --datasets mmfi \
  --protocol 1

# Copy to local SSD
rm -rf "$LOCAL_GOLD"
mkdir -p "$LOCAL_GOLD"
cp -a "$NFS_PROTO1/mmfi" "$LOCAL_GOLD/mmfi"
du -sh "$LOCAL_GOLD"
echo ">>> Data copy done at $(date)"

cd "$PROJECT_ROOT/ml/rfpose/training"

torchrun --standalone --nproc_per_node=4 train_wimose_diff.py \
  --config-name wimose_mmfi17j_diff_eagle \
  data.gold_dir="$LOCAL_GOLD" \
  training.checkpoint_dir="$CKPT_DIR" \
  "$ENCODER_ARG" \
  training.batch_size=32 \
  training.num_workers=4 \
  training.lr=5e-5

echo "=== Done $(date) ==="
