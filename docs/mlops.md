# MLOps Guide

## Dataset lifecycle
```text
Bronze: immutable raw CSI/events/video/health
Silver: decoded and validated frames
Gold: windowed ML-ready train/val/test dataset
```
Every Gold dataset must include manifest.json, stats.json, normalization.json, quality report, source session list, preprocess version, and teacher version.

## Training lifecycle
```text
Create dataset_version → create training_job → submit Helios Slurm → train/checkpoint → evaluate → export ONNX → package model → register candidate → eval gate → promote/archive
```

## Helios rules
Helios is batch compute, not service hosting. Use `plgrid-gpu-gh200`, assume aarch64 on GH200 nodes, checkpoint before the 48h limit, upload artifacts to S3/MLflow, and never trust `$SCRATCH` as source of truth.

## Eval gates
Minimum gates: macro_f1, latency_p95, empty_false_positive, node_dropout_test, temporal_jitter.

## Hard-case loop
Capture low-confidence, high-drift, node-disagreement, pose-jitter, and false-positive windows for adapter fine-tuning.
