#!/bin/bash
#SBATCH --job-name=wimose-action-only
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
NFS_PROTO1="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2-proto1"
LOCAL_GOLD="/tmp/rfpose_gold_mmfi_proto1_${SLURM_JOB_ID}"
POSE_CKPT="$PROJECT_ROOT/checkpoints/wimose-mmfi17j-proto1-v1/best.pt"
CKPT_DIR="$PROJECT_ROOT/checkpoints/wimose-mmfi17j-proto1-action-only-v1"

mkdir -p "$PROJECT_ROOT/logs" "$CKPT_DIR"
export MLFLOW_TRACKING_URI="http://207.180.243.242:5000"
export MLFLOW_S3_ENDPOINT_URL="http://207.180.243.242:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin123"

echo "=== WiMose Proto1 action-only | Job $SLURM_JOB_ID | $(date) ==="
test -f "$POSE_CKPT" || { echo "Missing pose checkpoint: $POSE_CKPT"; exit 1; }

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/ml:${PYTHONPATH:-}"
pip install -q -e "$PROJECT_ROOT/ml" 2>&1 | tail -1

python3 "$PROJECT_ROOT/scripts/assign_subject_splits.py" \
  --gold-root "$NFS_PROTO1" --datasets mmfi --protocol 1

rm -rf "$LOCAL_GOLD" && mkdir -p "$LOCAL_GOLD"
cp -a "$NFS_PROTO1/mmfi" "$LOCAL_GOLD/mmfi"

cd "$PROJECT_ROOT/ml/rfpose/training"
python train_wimose.py \
  --config-name wimose_mmfi17j_proto1_action_only_eagle \
  data.gold_dir="$LOCAL_GOLD" \
  training.pretrained_from="$POSE_CKPT" \
  training.checkpoint_dir="$CKPT_DIR" \
  training.batch_size=32 \
  training.num_workers=2

echo "=== Done $(date) ==="
