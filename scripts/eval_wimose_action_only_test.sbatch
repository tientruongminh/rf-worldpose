#!/bin/bash
#SBATCH --job-name=eval-action-only
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err
# Usage:
#   sbatch --dependency=afterok:JOBID scripts/eval_wimose_action_only_test.sbatch

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
GOLD_DIR="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2-proto1"
CKPT="${CKPT:-$PROJECT_ROOT/checkpoints/wimose-mmfi17j-proto1-action-only-v1/best_action.pt}"
OUT="$PROJECT_ROOT/checkpoints/wimose-mmfi17j-proto1-action-only-v1/eval_test.json"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/ml:${PYTHONPATH:-}"
cd "$PROJECT_ROOT/ml"

test -f "$CKPT" || { echo "Checkpoint not found: $CKPT"; exit 1; }

python -m rfpose.evaluation.eval_wimose_action \
  --checkpoint "$CKPT" \
  --gold-dir "$GOLD_DIR" \
  --datasets mmfi \
  --batch-size 64 \
  --output "$OUT"

echo "=== Eval done → $OUT ==="
python3 -c "import json; r=json.load(open('$OUT')); print('Action acc:', round(r.get('action_accuracy',0),4)); print('Macro F1:', round(r.get('action_macro_f1',0),4)); print('N:', r.get('n_samples'))"
