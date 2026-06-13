#!/bin/bash
#SBATCH --job-name=wimose-wipose18j
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/wimose_wipose18j_%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/wimose_wipose18j_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

# ── environment ──────────────────────────────────────────────────────────────
source /home/tiencd/rf-worldpose/.venv/bin/activate
export PYTHONPATH=/home/tiencd/rf-worldpose/ml:$PYTHONPATH

# ── log dir ──────────────────────────────────────────────────────────────────
mkdir -p /mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs
mkdir -p /mnt/storage_6/project_data/pl0501-01/rf-worldpose/checkpoints/wimose-wipose18j-v1

echo "=== Wi-Mose WiPose 18j | Job $SLURM_JOB_ID | $(date) ==="
echo "Node: $(hostname)  GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── torchrun DDP: 2 processes on 2 GPUs ──────────────────────────────────────
cd /home/tiencd/rf-worldpose/ml/rfpose/training

torchrun \
    --standalone \
    --nproc_per_node=2 \
    train_wimose.py \
    --config-name wimose_wipose18j_eagle

echo "=== Done $(date) ==="
