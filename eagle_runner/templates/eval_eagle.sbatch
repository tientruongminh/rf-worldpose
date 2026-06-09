#!/bin/bash
#SBATCH --job-name=rfpose-eval-{{ job_id }}
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err

set -eo pipefail

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
date

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
ML_DIR="$PROJECT_ROOT/ml"
GOLD_DIR="$PROJECT_ROOT/data/gold/rfpose-unified-v2"

export PIP_CACHE_DIR="$PROJECT_ROOT/.pip-cache"
export TMPDIR="$PROJECT_ROOT/.tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"

source "$PROJECT_ROOT/.venv/bin/activate"
pip install --no-cache-dir -e "$ML_DIR" 2>&1 | tail -1
pip install --no-cache-dir scikit-learn 2>&1 | tail -1

export MLFLOW_S3_ENDPOINT_URL="http://207.180.243.242:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin123"

cd "$ML_DIR"

echo "=== Evaluating models ==="

# Eval rootrel-wipose (best overall: 286.1mm)
echo "--- rootrel-wipose ---"
python -m rfpose.evaluation.eval_v2 \
    --s3-path s3://rfpose/mlflow/4/be52da0cebdb4e8e9c6cfa33bda591ca/artifacts/best.pt \
    --gold-dir "$GOLD_DIR" \
    --variant rootrel \
    --output /tmp/eval_rootrel_wipose.json \
    --batch-size 64

# Eval metafi-mmfi (317.3mm, baseline comparison)
echo "--- metafi-mmfi ---"
python -m rfpose.evaluation.eval_v2 \
    --s3-path s3://rfpose/mlflow/4/4880a549c9494b3999052afb1300d3f2/artifacts/best.pt \
    --gold-dir "$GOLD_DIR" \
    --variant metafi \
    --output /tmp/eval_metafi_mmfi.json \
    --batch-size 64

# Eval mmfi-only (baseline: 327.4mm)
echo "--- mmfi-only-base ---"
python -m rfpose.evaluation.eval_v2 \
    --s3-path s3://rfpose/mlflow/4/dc309454d4504b72b8118bd80a1dcbbb/artifacts/best.pt \
    --gold-dir "$GOLD_DIR" \
    --variant base \
    --output /tmp/eval_mmfi_only.json \
    --batch-size 64

# Eval rootrel-combined (318.8mm)
echo "--- rootrel-combined ---"
python -m rfpose.evaluation.eval_v2 \
    --s3-path s3://rfpose/mlflow/4/54d03a07d592476f8863b9636c680e26/artifacts/best.pt \
    --gold-dir "$GOLD_DIR" \
    --variant rootrel \
    --output /tmp/eval_rootrel_combined.json \
    --batch-size 64

echo "=== All evaluations complete ==="
date
