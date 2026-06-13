#!/usr/bin/env bash
# Submit RF-WorldPose training job to Eagle HPC from VPS.
#
# Usage (run on VPS2 where SSH to Eagle is configured):
#   ./scripts/eagle_submit_train.sh                                  # default: WiMose Proto1 pose
#   LIST_CONFIGS=1 ./scripts/eagle_submit_train.sh                   # show registered model presets
#   CONFIG=rootrel_mmfi_eagle ./scripts/eagle_submit_train.sh        # RootRel multitask
#   CONFIG=ssl_pretrain ./scripts/eagle_submit_train.sh              # SSL pretrain
#   SKIP_DATA=1 ./scripts/eagle_submit_train.sh                     # skip 24GB gold rsync
#   DRY_RUN=1 ./scripts/eagle_submit_train.sh                       # render sbatch only
#   SMOKE=1 ./scripts/eagle_submit_train.sh                          # 1 epoch smoke test
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
EAGLE_HOST="${EAGLE_HOST:-eagle}"
EAGLE_ROOT="${EAGLE_ROOT:-pl0501-01/project_data/rf-worldpose}"
CONFIG="${CONFIG:-wimose_mmfi17j_proto1_eagle}"
TRAIN_MODULE="${TRAIN_MODULE:-}"
JOB_ID="${JOB_ID:-rfpose-$(date +%Y%m%d-%H%M%S)}"
MLFLOW_URI="${MLFLOW_URI:-http://207.180.243.242:5000}"

echo "=== Eagle train submit ==="
echo "REPO:    $REPO_ROOT"
echo "EAGLE:   $EAGLE_HOST:$EAGLE_ROOT"
echo "JOB_ID:  $JOB_ID"
echo "CONFIG:  $CONFIG"
echo "MODULE:  ${TRAIN_MODULE:-(auto from config)}"

export PYTHONPATH="$REPO_ROOT/eagle_runner:${PYTHONPATH:-}"
if [ "${LIST_CONFIGS:-0}" = "1" ]; then
python3 - <<'PY'
from rfpose_eagle.registry import list_presets

print("Registered RF-WorldPose training presets:")
for p in list_presets():
    mark = " *" if p["recommended"] else "  "
    print(f"{mark} {p['config_name']:<45} {p['task']:<10} {p['model_family']:<22} {p['dataset_version']}")
PY
exit 0
fi

DATASET_VERSION="${DATASET_VERSION:-$(python3 - <<PY
from rfpose_eagle.registry import get_preset
print(get_preset("${CONFIG}").dataset_version)
PY
)}"
echo "DATASET: $DATASET_VERSION"

# --- 1. Sync code ---
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1 — skipping SSH/rsync; rendering sbatch locally"
else
  echo ">>> Syncing ml/ and eagle_runner/ ..."
  ssh "$EAGLE_HOST" "mkdir -p $EAGLE_ROOT/ml $EAGLE_ROOT/data/gold $EAGLE_ROOT/logs $EAGLE_ROOT/checkpoints"
  rsync -az --delete \
    --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    "$REPO_ROOT/ml/" "$EAGLE_HOST:$EAGLE_ROOT/ml/"
  rsync -az \
    "$REPO_ROOT/eagle_runner/" "$EAGLE_HOST:$EAGLE_ROOT/eagle_runner/"
fi

# --- 2. Sync gold data (optional skip) ---
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1 — skipping gold sync"
elif [ "${SKIP_DATA:-0}" != "1" ]; then
  if ssh "$EAGLE_HOST" "test -d $EAGLE_ROOT/data/gold/$DATASET_VERSION"; then
    echo ">>> Gold data already on Eagle — skipping rsync (set FORCE_DATA=1 to override)"
    if [ "${FORCE_DATA:-0}" = "1" ]; then
      echo ">>> FORCE_DATA=1 — re-syncing gold ..."
      rsync -az --info=progress2 \
        "$REPO_ROOT/data/gold/$DATASET_VERSION/" \
        "$EAGLE_HOST:$EAGLE_ROOT/data/gold/$DATASET_VERSION/"
    fi
  else
    if [ -d "$REPO_ROOT/data/gold/$DATASET_VERSION" ]; then
      echo ">>> Syncing gold data: $DATASET_VERSION ..."
      rsync -az --info=progress2 \
        "$REPO_ROOT/data/gold/$DATASET_VERSION/" \
        "$EAGLE_HOST:$EAGLE_ROOT/data/gold/$DATASET_VERSION/"
    else
      echo ">>> Local gold not found for $DATASET_VERSION — assuming it already exists on Eagle or will be mounted"
    fi
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

python3 - <<PY
from rfpose_eagle.submit import EagleJobSpec, submit_training_job
from rfpose_eagle.registry import get_preset

_preset = get_preset("${CONFIG}")
spec = EagleJobSpec(
    job_id="${JOB_ID}",
    config_name="${CONFIG}",
    train_module="${TRAIN_MODULE}",
    dataset_version="${DATASET_VERSION}" or _preset.dataset_version,
    gpus=int("${GPUS:-0}") or _preset.gpus,
    cpus=int("${CPUS:-0}") or _preset.cpus,
    mem="${MEM:-}" or _preset.mem,
    time_limit="${TIME_LIMIT:-}" or _preset.time_limit,
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
