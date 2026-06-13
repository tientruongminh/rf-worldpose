# RF-WorldPose

RF-WorldPose is a research and platform codebase for WiFi CSI human sensing. It
turns wireless channel state information into ML-ready datasets, trains pose and
action models on HPC, tracks experiments in MLflow, and exposes backend services
for dataset, training, model and inference operations.

The current research conclusion is:

- **Pose regression:** use the single-task **WiMose Proto1** pipeline on MM-Fi
  Protocol 1. Current held-out test result: **173.6 mm MPJPE / 157.1 mm
  PA-MPJPE** on 1,104 test samples.
- **Action classification:** use the **RootRel multitask/action** pipeline when
  action labels are the goal. RootRel reaches about **91% accuracy and macro-F1**
  on unified-v2, but its pose regression is weaker than WiMose.

Do not compare numbers across protocols without checking dataset version,
skeleton, split and preprocessing. The repo now uses `rfpose_eagle.registry` as
the canonical source for model/config submit presets.

## Architecture

```text
CSI / dataset sources
  -> Bronze / Silver / Gold ETL
  -> ML-ready memmap/parquet datasets
  -> PyTorch training on Eagle HPC via Slurm
  -> MLflow metrics + artifacts + checkpoints
  -> evaluation reports + thesis figures
  -> API control plane + inference service
```

| Layer | Path | Purpose |
| --- | --- | --- |
| ML models and training | `ml/rfpose/` | WiMose, RootRel, ViT2D, SSL, eval scripts |
| Training configs | `ml/configs/` | Hydra YAML configs used by Slurm jobs |
| HPC submitter | `eagle_runner/` | Canonical preset registry, sbatch rendering, submit |
| Submit scripts | `scripts/` | Local/VPS helpers for Slurm eval, viz, train jobs |
| API service | `services/api/` | Training jobs, configs, HPC status, datasets, models |
| Inference service | `services/inference/` | ONNX runtime API + NATS realtime inference |
| Data/ETL | `pipelines/`, `data/` | Bronze/Silver/Gold dataset pipeline |
| Reports and thesis | `docs/` | Research logs, thesis LaTeX, figures |

## Main Models

| Config | Task | Model | Dataset | Use when |
| --- | --- | --- | --- | --- |
| `wimose_mmfi17j_proto1_eagle` | pose | WiMoseNet Proto1 | `rfpose-humanlike-v2-proto1` | Main MM-Fi 17J pose result |
| `rootrel_mmfi_eagle` | multitask | CSITransformerPoseRootRel | `rfpose-unified-v2` | Pose/action multitask baseline |
| `rootrel_mmfi_action_only_from_scratch_eagle` | action | RootRel action | `rfpose-unified-v2` | Action-only ablation |
| `rootrel_mmfi_pose_only_eagle` | pose | RootRel pose | `rfpose-unified-v2` | Transformer pose ablation |
| `vit2d_mmfi_eagle` | pose | CSIViT2DPose | `rfpose-unified-v2` | ViT/attention baseline |
| `ssl_eagle` | pretrain | CSI encoder SSL | `rfpose-unified-v2` | Representation pretraining |

List the registered presets:

```bash
LIST_CONFIGS=1 ./scripts/eagle_submit_train.sh
```

The registry lives in:

```text
eagle_runner/rfpose_eagle/registry.py
```

Add new models there first, then add the matching Hydra config in `ml/configs/`.

## Results Snapshot

### Pose

| Model | Split | N | MPJPE | PA-MPJPE | Protocol |
| --- | ---: | ---: | ---: | ---: | --- |
| WiMose Proto1 | val | 1,500 | 157.3 mm | 147.5 mm | MM-Fi Proto1 |
| WiMose Proto1 | test | 1,104 | 173.6 mm | 157.1 mm | MM-Fi Proto1 |
| RootRel pose | val | 1,296 | 306.6 mm | 216.4 mm | unified-v2 |
| RootRel pose | test | 1,296 | 310.5 mm | 216.3 mm | unified-v2 |

### Action

| Model | Split | Accuracy | Macro-F1 | Protocol |
| --- | ---: | ---: | ---: | --- |
| RootRel multitask/action | val | 91.28% | 90.95% | unified-v2 |
| RootRel multitask/action | test | 91.51% | 91.37% | unified-v2 |
| Proto1 action head | val/test | 18-19% | low | Proto1 |

## Local Development

### Requirements

- Python 3.11+
- Docker + Docker Compose
- PostgreSQL client tools if running migrations manually
- SSH key configured for Eagle if submitting HPC jobs
- Optional: CUDA/PyTorch environment for local ML smoke tests

### Start the dev stack

```bash
cp .env.example .env
./scripts/dev_up.sh
```

Useful local URLs:

| Service | URL |
| --- | --- |
| API Swagger | http://localhost:8080/docs |
| MLflow | http://localhost:5000 |
| MinIO console | http://localhost:9003 |
| Dagster | http://localhost:3004 |
| Grafana | http://localhost:3002 |
| Prometheus | http://localhost:9090 |

Stop services:

```bash
docker compose -f infra/docker-compose/docker-compose.yml --env-file .env down
```

Reset local state:

```bash
docker compose -f infra/docker-compose/docker-compose.yml --env-file .env down -v
```

## Submit Training Jobs

### Recommended pose job

```bash
./scripts/eagle_submit_train.sh
```

Default config is now:

```text
wimose_mmfi17j_proto1_eagle
```

This is the main single-task pose model.

### RootRel multitask/action

```bash
CONFIG=rootrel_mmfi_eagle ./scripts/eagle_submit_train.sh
```

### RootRel action from scratch

```bash
CONFIG=rootrel_mmfi_action_only_from_scratch_eagle ./scripts/eagle_submit_train.sh
```

### Dry run sbatch render

```bash
DRY_RUN=1 CONFIG=wimose_mmfi17j_proto1_eagle ./scripts/eagle_submit_train.sh
```

### Smoke test

```bash
SMOKE=1 CONFIG=quick_test ./scripts/eagle_submit_train.sh
```

### Common submit environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONFIG` | `wimose_mmfi17j_proto1_eagle` | Registered training config |
| `JOB_ID` | timestamped `rfpose-*` | Slurm/checkpoint run id |
| `DATASET_VERSION` | preset default | Gold dataset version |
| `GPUS` | preset default | GPU count in rendered sbatch |
| `CPUS` | preset default | CPU count |
| `MEM` | preset default | Slurm memory request |
| `TIME_LIMIT` | preset default | Slurm wall time |
| `SKIP_DATA` | `0` | Do not rsync local gold data |
| `FORCE_DATA` | `0` | Re-sync gold data even if remote exists |
| `DRY_RUN` | `0` | Render sbatch without submitting |

## Evaluate Models

Run the SOTA/internal comparison job on Eagle:

```bash
sbatch scripts/eval_sota_pose_compare.sbatch
```

Outputs:

```text
eval_results/sota-pose-compare/
  wimose-proto1-val.json
  wimose-proto1-test.json
  rootrel-mmfi-v1-val-pose.json
  rootrel-mmfi-v1-test-pose.json
  summary.json
```

Generate WiMose top-k visualizations:

```bash
sbatch scripts/viz_wimose_proto1_top.sbatch
```

Export MLflow curves for thesis/report figures:

```bash
python scripts/fetch_mlflow_training_curves.py
```

## API Usage

The FastAPI service exposes control-plane endpoints.

### List training presets

```bash
curl http://localhost:8080/api/v1/hpc/configs
```

### Create a job record

```bash
curl -X POST http://localhost:8080/api/v1/training-jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "wimose-proto1-report-run",
    "dataset_version": "rfpose-humanlike-v2-proto1",
    "train_config": "wimose_mmfi17j_proto1_eagle",
    "submitted_by": "tiencd"
  }'
```

### Dry-run submit

```bash
curl -X POST 'http://localhost:8080/api/v1/hpc/training-jobs/wimose-proto1-report-run/submit?dry_run=true'
```

### Submit to Eagle

```bash
curl -X POST http://localhost:8080/api/v1/hpc/training-jobs/wimose-proto1-report-run/submit
```

### Refresh/cancel/logs

```bash
curl -X POST http://localhost:8080/api/v1/hpc/training-jobs/wimose-proto1-report-run/refresh-status
curl http://localhost:8080/api/v1/hpc/training-jobs/wimose-proto1-report-run/logs
curl -X POST http://localhost:8080/api/v1/hpc/training-jobs/wimose-proto1-report-run/cancel
```

## Inference Service

The inference service loads:

```text
$MODEL_DIR/model.onnx
```

Endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness and model loaded state |
| `GET /status` | model IO shapes, NATS, buffer and stats |
| `POST /predict` | manual CSI window inference |
| `POST /reload` | reload `model.onnx` from disk |
| `GET /predictions/recent` | recent NATS realtime predictions |

Manual prediction payload:

```json
{
  "csi": [[[0.1, 0.2], [0.3, 0.4]]]
}
```

The service now decodes both action logits and pose outputs. Pose outputs with
shape `(B, J, 3)` are returned as `pose_3d`; `(B, J, 2)` is returned as `pose_2d`.

## Thesis/Report Artifacts

Key generated artifacts:

```text
viz_output/mlflow_curves/
viz_output/wimose_proto1_top/
viz_output/mmfi17j_best/
eval_results/sota-pose-compare/
docs/thesis/
```

Important thesis files currently prepared:

```text
docs/thesis/chapter2_co_so_ly_thuyet.tex
docs/thesis/chapter4_3_to_4_6_results_expanded.tex
```

## Development Checks

Compile Python:

```bash
python -m compileall eagle_runner services/api/src services/inference/src ml/rfpose
```

Run API contract tests:

```bash
PYTHONPATH=services/api/src:eagle_runner pytest services/api/tests
```

Render a submit dry run:

```bash
DRY_RUN=1 SKIP_DATA=1 ./scripts/eagle_submit_train.sh
```

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Wrong model submitted | Config not in registry | Add/update `eagle_runner/rfpose_eagle/registry.py` |
| Pose visualization is far away or collapsed | Coordinate mismatch | Check `center_pose`, `root_joint`, CSI mean/std metadata |
| Proto1 action stuck at 18-19% | Majority baseline behavior | Use RootRel/unified-v2 for action, or redesign Proto1 action training |
| Slurm shows FAILED after checkpoint exists | DDP teardown or post-train failure | Check `best.pt`, eval checkpoint before resubmit |
| MLflow logging too slow | Logging per batch | Log metrics per epoch for long jobs |
| Inference says no model loaded | Missing `model.onnx` | Copy/export model to `$MODEL_DIR/model.onnx`, then `POST /reload` |

## Current Research Position

Use this wording in reports:

1. **WiMose Proto1 is the strongest pose model in the RF-WorldPose pipeline** and
   beats the reproduced/internal baselines under the MM-Fi Protocol 1 setup.
2. **Single-task training is better for pose regression** in the current
   experiments. **Multitask training is better suited for action classification**
   because pose/motion supervision enriches the action representation.

Avoid claiming a global SOTA result unless the comparison uses the exact same
dataset, skeleton, split, preprocessing and evaluation protocol as the external
paper.
