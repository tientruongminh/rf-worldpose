#!/usr/bin/env bash
# Submit RF-WorldPose training job to Eagle HPC from VPS.
#
# Usage (run on VPS2 where SSH to Eagle is configured):
#   ./scripts/eagle_submit_train.sh                                  # default: transformer_eagle
#   CONFIG=ssl_pretrain ./scripts/eagle_submit_train.sh              # SSL pretrain
#   CONFIG=finetune_room ./scripts/eagle_submit_train.sh             # fine-tune
#   SKIP_DATA=1 ./scripts/eagle_submit_train.sh                     # skip 24GB gold rsync
#   DRY_RUN=1 ./scripts/eagle_submit_train.sh                       # render sbatch only
#   SMOKE=1 ./scripts/eagle_submit_train.sh                          # 1 epoch smoke test
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/Ngan/rf-worldpose}"
EAGLE_HOST="${EAGLE_HOST:-eagle}"
EAGLE_ROOT="${EAGLE_ROOT:-pl0501-01/project_data/rf-worldpose}"
CONFIG="${CONFIG:-transformer_eagle}"
TRAIN_MODULE="${TRAIN_MODULE:-}"
JOB_ID="${JOB_ID:-rfpose-$(date +%Y%m%d-%H%M%S)}"
MLFLOW_URI="${MLFLOW_URI:-http://207.180.243.242:5000}"

echo "=== Eagle train submit ==="
echo "REPO:    $REPO_ROOT"
echo "EAGLE:   $EAGLE_HOST:$EAGLE_ROOT"
echo "JOB_ID:  $JOB_ID"
echo "CONFIG:  $CONFIG"
echo "MODULE:  ${TRAIN_MODULE:-(auto from config)}"

# --- 1. Sync code ---
echo ">>> Syncing ml/ and eagle_runner/ ..."
ssh "$EAGLE_HOST" "mkdir -p $EAGLE_ROOT/ml $EAGLE_ROOT/data/gold $EAGLE_ROOT/logs $EAGLE_ROOT/checkpoints"
rsync -az --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
  "$REPO_ROOT/ml/" "$EAGLE_HOST:$EAGLE_ROOT/ml/"
rsync -az \
  "$REPO_ROOT/eagle_runner/" "$EAGLE_HOST:$EAGLE_ROOT/eagle_runner/"

# --- 2. Sync gold data (optional skip) ---
if [ "${SKIP_DATA:-0}" != "1" ]; then
  if ssh "$EAGLE_HOST" "test -f $EAGLE_ROOT/data/gold/rfpose-multitask-v1/uthar/x.npy"; then
    echo ">>> Gold data already on Eagle — skipping rsync (set FORCE_DATA=1 to override)"
    if [ "${FORCE_DATA:-0}" = "1" ]; then
      echo ">>> FORCE_DATA=1 — re-syncing gold ..."
      rsync -az --info=progress2 \
        "$REPO_ROOT/data/gold/rfpose-multitask-v1/" \
        "$EAGLE_HOST:$EAGLE_ROOT/data/gold/rfpose-multitask-v1/"
    fi
  else
    echo ">>> Syncing gold data (~24 GB, may take a while) ..."
    rsync -az --info=progress2 \
      "$REPO_ROOT/data/gold/rfpose-multitask-v1/" \
      "$EAGLE_HOST:$EAGLE_ROOT/data/gold/rfpose-multitask-v1/"
  fi
else
  echo ">>> SKIP_DATA=1 — not syncing gold"
fi

# --- 3. Submit Slurm job ---
EPOCHS=50
BATCH_SIZE=32
TRAIN_DRY_RUN_PY="False"
if [ "${SMOKE:-0}" = "1" ]; then
  EPOCHS=1
  BATCH_SIZE=2
  TRAIN_DRY_RUN_PY="True"
  echo ">>> SMOKE mode: 1 epoch, dry_run=true"
fi

SBATCH_DRY_PY="False"
if [ "${DRY_RUN:-0}" = "1" ]; then
  SBATCH_DRY_PY="True"
fi

export PYTHONPATH="$REPO_ROOT/eagle_runner:${PYTHONPATH:-}"
python3 - <<PY
from rfpose_eagle.submit import EagleJobSpec, submit_training_job

spec = EagleJobSpec(
    job_id="${JOB_ID}",
    config_name="${CONFIG}",
    train_module="${TRAIN_MODULE}",
    mlflow_tracking_uri="${MLFLOW_URI}",
    epochs=${EPOCHS},
    batch_size=${BATCH_SIZE},
    dry_run=${TRAIN_DRY_RUN_PY},
)
result = submit_training_job(spec, ssh_host="${EAGLE_HOST}", sync=False, dry_run=${SBATCH_DRY_PY})
if ${SBATCH_DRY_PY}:
    print("=== DRY RUN sbatch ===")
    print(result)
else:
    print(f"Submitted Slurm job: {result}")
    print(f"Monitor: ssh ${EAGLE_HOST} squeue -u \$(whoami)")
    print(f"Logs:    ssh ${EAGLE_HOST} tail -f ${EAGLE_ROOT}/logs/rfpose-{result}.out")
PY

echo "=== Done ==="
