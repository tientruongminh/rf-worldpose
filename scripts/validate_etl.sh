#!/usr/bin/env bash
set -euo pipefail
python3 pipelines/dagster/rfpose_pipelines/etl/bronze_to_silver.py --bronze-root /tmp/rfpose-bronze --silver-out /tmp/rfpose-silver/csi_decoded.parquet
python3 pipelines/dagster/rfpose_pipelines/etl/silver_to_gold.py --silver-path /tmp/rfpose-silver/csi_decoded.parquet --gold-dir /tmp/rfpose-gold
PYTHONPATH=ml python3 -m rfpose.training.train --dataset /tmp/rfpose-gold --epochs 1 --batch-size 4 --output /tmp/rfpose-etl-train
PYTHONPATH=ml python3 -m rfpose.evaluation.eval --checkpoint /tmp/rfpose-etl-train/best.pt --dataset /tmp/rfpose-gold --output /tmp/rfpose-etl-eval.json
PYTHONPATH=ml python3 -m rfpose.evaluation.eval_gate --report /tmp/rfpose-etl-eval.json --min-macro-f1 0.0
