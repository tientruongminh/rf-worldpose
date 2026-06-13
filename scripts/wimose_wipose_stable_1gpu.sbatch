#!/bin/bash
#SBATCH --job-name=wimose-wipose-stable
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
SRC_GOLD="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2"
LOCAL_GOLD="/tmp/rfpose_gold_wipose_stable_${SLURM_JOB_ID}"
CKPT_DIR="$PROJECT_ROOT/checkpoints/wimose-wipose18j-stable-v1"

mkdir -p "$PROJECT_ROOT/logs" "$CKPT_DIR"
export MLFLOW_TRACKING_URI="http://207.180.243.242:5000"
export MLFLOW_S3_ENDPOINT_URL="http://207.180.243.242:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin123"
export PYTHONPATH="$PROJECT_ROOT/ml:${PYTHONPATH:-}"

echo "=== WiMose WiPose stable 1GPU | Job $SLURM_JOB_ID | $(date) ==="
echo "Node: $(hostname) GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

rm -rf "$LOCAL_GOLD"
mkdir -p "$LOCAL_GOLD"
cp -a "$SRC_GOLD/wipose" "$LOCAL_GOLD/wipose"
du -sh "$LOCAL_GOLD"
python3 -m json.tool "$LOCAL_GOLD/wipose/manifest.json" | sed -n '1,45p'

source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT/ml/rfpose/training"

python train_wimose.py \
  --config-name wimose_wipose18j_eagle \
  data.gold_dir="$LOCAL_GOLD" \
  training.checkpoint_dir="$CKPT_DIR" \
  training.batch_size=16 \
  training.num_workers=0 \
  mlflow.experiment_name="rf-worldpose-wimose" \
  mlflow.run_name="wimose-wipose18j-stable-1gpu-v1"

echo "=== Done $(date) ==="
