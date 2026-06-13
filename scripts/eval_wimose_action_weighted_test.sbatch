#!/bin/bash
#SBATCH --job-name=eval-action-wt
#SBATCH --partition=tesla
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.out
#SBATCH --error=/mnt/storage_6/project_data/pl0501-01/rf-worldpose/logs/%x-%j.err
# Usage: sbatch --dependency=afterok:<TRAIN_JOB> scripts/eval_wimose_action_weighted_test.sbatch

set -euo pipefail

PROJECT_ROOT="/mnt/storage_6/project_data/pl0501-01/rf-worldpose"
VENV_DIR="$PROJECT_ROOT/.venv"
GOLD_DIR="$PROJECT_ROOT/data/gold/rfpose-humanlike-v2-proto1"
CKPT_DIR="$PROJECT_ROOT/checkpoints/wimose-mmfi17j-proto1-action-weighted-v1"
CKPT="${CKPT:-$CKPT_DIR/best_action.pt}"
OUT="$CKPT_DIR/eval_test.json"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/ml:${PYTHONPATH:-}"
cd "$PROJECT_ROOT/ml"

if [ ! -f "$CKPT" ]; then
  echo "best_action.pt missing, falling back to best.pt"
  CKPT="$CKPT_DIR/best.pt"
fi
test -f "$CKPT" || { echo "Checkpoint not found: $CKPT"; exit 1; }

python -m rfpose.evaluation.eval_wimose_action \
  --checkpoint "$CKPT" \
  --gold-dir "$GOLD_DIR" \
  --datasets mmfi \
  --batch-size 64 \
  --output "$OUT"

echo "=== Eval done → $OUT ==="
cat "$OUT" | python3 -c "import sys,json; r=json.load(sys.stdin); print('MPJPE mm:', round(r['mpjpe_mm'],1)); print('Action acc:', round(r.get('action_accuracy',0),3) if 'action_accuracy' in r else 'n/a')"
