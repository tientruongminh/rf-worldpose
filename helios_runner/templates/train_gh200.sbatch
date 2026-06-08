#!/bin/bash -l
#SBATCH --job-name=rfpose-train
#SBATCH --partition={{ partition }}
#SBATCH --account={{ account }}
#SBATCH --time={{ time_limit | default("24:00:00") }}
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "===== RF-WorldPose Training ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
date

module purge
module load Python/3.11 CUDA/12.2

export RUN_ID="${RUN_ID:-rfpose-${SLURM_JOB_ID}}"
export DATASET_VERSION="{{ dataset_version }}"
export TRAIN_CONFIG="{{ train_config }}"
export S3_BUCKET="{{ s3_bucket }}"
export S3_ENDPOINT_URL="{{ s3_endpoint_url }}"
export MLFLOW_TRACKING_URI="{{ mlflow_tracking_uri }}"

WORKDIR="${SCRATCH:-$PWD}/rfpose/runs/${RUN_ID}"
mkdir -p "$WORKDIR" logs
cd "$WORKDIR"

# Clone repo and checkout branch
if [ ! -d "repo" ]; then
    git clone --depth 1 --branch {{ git_branch }} {{ git_repo }} repo
else
    cd repo && git pull origin {{ git_branch }} && cd ..
fi

# Setup venv
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e repo/ 2>/dev/null || true

# Sync dataset from S3
aws s3 sync "s3://${S3_BUCKET}/gold/${DATASET_VERSION}/" ./dataset/ --endpoint-url "$S3_ENDPOINT_URL"

echo "===== Starting training: {{ script_path }} ====="

# Parse hyperparams as env vars
{{ train_config }}

# Run training script
python "repo/{{ script_path }}" \
    --dataset ./dataset \
    --output ./outputs \
    --run-id "$RUN_ID"

# Eval + export
if [ -f "./outputs/best.pt" ]; then
    python -m rfpose.evaluation.eval --checkpoint ./outputs/best.pt --dataset ./dataset --output ./outputs/eval_report.json 2>/dev/null || true
    python -m rfpose.export.onnx --checkpoint ./outputs/best.pt --output ./outputs/model.onnx 2>/dev/null || true
fi

# Upload results
aws s3 sync ./outputs "s3://${S3_BUCKET}/models/${RUN_ID}/" --endpoint-url "$S3_ENDPOINT_URL"

echo "===== Training complete ====="
date
