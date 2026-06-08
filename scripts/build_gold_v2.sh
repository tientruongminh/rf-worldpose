#!/usr/bin/env bash
# Build rfpose-multitask-v2 by converting 3 remaining bronze datasets
# then merging with existing v1 datasets (uthar, wiar, wimans).
#
# Usage: bash scripts/build_gold_v2.sh [--bronze /path/to/bronze] [--gold /path/to/gold]

set -eo pipefail

BRONZE_ROOT="${1:-/app/data/bronze}"
GOLD_V1="${2:-/app/data/gold/rfpose-multitask-v1}"
GOLD_V2="${3:-/app/data/gold/rfpose-multitask-v2}"

echo "=== Build Gold v2 ==="
echo "Bronze: $BRONZE_ROOT"
echo "Gold v1: $GOLD_V1"
echo "Gold v2: $GOLD_V2"

mkdir -p "$GOLD_V2"

# Step 1: Convert new datasets
echo ""
echo "--- Step 1: Convert MMFi, Wi-Pose, wifipose ---"
python3 -m rfpose_pipelines.etl.bronze_to_gold_adapters \
    --bronze-root "$BRONZE_ROOT" \
    --gold-root "$GOLD_V2" \
    --dataset all

# Step 2: Copy existing datasets from v1
echo ""
echo "--- Step 2: Copy existing v1 datasets (uthar, wiar, wimans) ---"
for ds in uthar wiar wimans; do
    if [ -d "$GOLD_V1/$ds" ]; then
        echo "  Linking $ds from v1..."
        ln -sfn "$(realpath "$GOLD_V1/$ds")" "$GOLD_V2/$ds"
    fi
done

# Step 3: Build unified summary
echo ""
echo "--- Step 3: Build unified summary ---"
python3 -c "
import json, os, numpy as np
from pathlib import Path

gold = Path('$GOLD_V2')
datasets = {}
label_maps = {'activity': {}, 'environment': {}, 'location': {}, 'subject': {}}

for ds_dir in sorted(gold.iterdir()):
    if not ds_dir.is_dir():
        continue
    x_path = ds_dir / 'x.npy'
    if not x_path.exists():
        continue
    x = np.load(str(x_path), mmap_mode='r')
    manifest = {}
    mp = ds_dir / 'manifest.json'
    if mp.exists():
        manifest = json.loads(mp.read_text())

    lm_path = ds_dir / 'label_maps.json'
    if lm_path.exists():
        lm = json.loads(lm_path.read_text())
        for cat in label_maps:
            if cat in lm:
                label_maps[cat].update(lm[cat])

    datasets[ds_dir.name] = manifest or {
        'dataset': ds_dir.name,
        'num_samples': int(x.shape[0]),
        'x_shape': list(x.shape),
    }

total = sum(d.get('num_samples', 0) for d in datasets.values())
summary = {
    'datasets': datasets,
    'num_datasets': len(datasets),
    'num_samples': total,
    'window_frames': 60,
    'stride': 21,
    'label_maps': 'label_maps.json',
    'artifact_uri': str(gold),
}
(gold / 'summary.json').write_text(json.dumps(summary, indent=2))
(gold / 'label_maps.json').write_text(json.dumps(label_maps, indent=2))
print(f'Gold v2: {len(datasets)} datasets, {total} total windows')
for name, info in datasets.items():
    print(f'  {name}: {info.get(\"num_samples\", \"?\")} windows')
"

echo ""
echo "=== Done ==="
