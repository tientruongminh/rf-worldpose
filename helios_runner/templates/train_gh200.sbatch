#!/bin/bash
#SBATCH --job-name=rfpose-train
#SBATCH --partition={{ partition }}
#SBATCH --account={{ account }}
#SBATCH --time={{ time_limit | default("24:00:00") }}
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=480G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
echo "ARCH=$(uname -m)"
date

module purge
# TODO: module load proper aarch64/Rocky9 CUDA/PyTorch stack on GH200 compute node.

export RUN_ID="${RUN_ID:-rfpose-${SLURM_JOB_ID}}"
export DATASET_VERSION="{{ dataset_version }}"
export TRAIN_CONFIG="{{ train_config }}"
export S3_BUCKET="{{ s3_bucket }}"
export S3_ENDPOINT_URL="{{ s3_endpoint_url }}"
export MLFLOW_TRACKING_URI="{{ mlflow_tracking_uri }}"

WORKDIR="${SCRATCH:-$PWD}/rfpose/runs/${RUN_ID}"
mkdir -p "$WORKDIR" logs
cd "$WORKDIR"

aws s3 sync "s3://${S3_BUCKET}/gold/${DATASET_VERSION}/" ./dataset/ --endpoint-url "$S3_ENDPOINT_URL"

torchrun --nproc_per_node=4 -m rfpose.training.train \
  dataset.path=./dataset \
  run.id="$RUN_ID" \
  --config-name "$TRAIN_CONFIG"

python -m rfpose.evaluation.eval --run-id "$RUN_ID" --output ./outputs/eval_report.json
python -m rfpose.export.onnx --run-id "$RUN_ID" --output ./outputs/model.onnx
aws s3 sync ./outputs "s3://${S3_BUCKET}/models/${RUN_ID}/" --endpoint-url "$S3_ENDPOINT_URL"
