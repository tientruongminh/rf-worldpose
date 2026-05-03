# RF-WorldPose Platform — Final Production/Research Architecture

## Executive verdict

Kiến trúc này **đủ chuẩn production/research** cho một hệ WiFi DensePose/RF sensing nghiêm túc: tách rõ data plane, ML plane, serving plane và ops/security plane; tận dụng đúng Helios GH200 như batch training backend; giữ edge/VPS làm nơi collection/inference/ops ổn định; có data lineage, model registry, eval gate, rollback và feedback loop.

Điểm mạnh nhất của kiến trúc là không nhầm GH200/HPC thành production server. Helios chỉ chạy job train/eval/export qua Slurm, còn toàn bộ source of truth nằm ở MinIO/S3 + Postgres + MLflow bên ngoài. Đây là thiết kế đúng cho môi trường HPC có time limit, architecture ARM/aarch64 và storage scratch không bền vững.

## System name

```text
RF-WorldPose Platform
```

## Mission

```text
Thu CSI từ ESP32-S3
→ build dataset chuẩn
→ train model trên Helios GH200
→ deploy WiFi-only skeleton/DensePose inference
→ monitor + feedback + cải thiện liên tục
```

## Final end-to-end architecture

```text
ESP32-S3 CSI Mesh
  → Rust Edge Gateway
  → NATS/Ingest
  → MinIO/S3 Data Lake
  → Dagster ETL Bronze/Silver/Gold
  → Dataset Registry
  → Helios GH200 Slurm Training
  → MLflow Model Registry
  → ONNX Runtime Edge / Triton-TensorRT Cloud Deployment
  → Monitoring + Security + OTA + Rollback + Feedback Loop
```

```text
┌──────────────────────────────────────────────┐
│ ESP32-S3 CSI Nodes                            │
│ 4 nodes, ESP-IDF firmware                     │
└──────────────────┬───────────────────────────┘
                   │ UDP binary CSI packets
                   ▼
┌──────────────────────────────────────────────┐
│ Edge Gateway                                  │
│ Rust/Tokio                                    │
│ receive, validate, buffer, local inference    │
└──────────────────┬───────────────────────────┘
                   │ NATS / batch upload
                   ▼
┌──────────────────────────────────────────────┐
│ Cloud/VPS Control Plane                       │
│ API, Ingest, Metadata, Dashboard              │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Data Lake + ETL                               │
│ MinIO/S3 + Dagster + Postgres                 │
│ Bronze → Silver → Gold                        │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Dataset Registry                              │
│ versioned datasets + quality reports          │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Helios GH200 Training Backend                 │
│ Slurm plgrid-gpu-gh200                        │
│ train/eval/export                             │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Model Registry                                │
│ MLflow + artifacts + eval gates               │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Deployment                                    │
│ Edge ONNX Runtime / Cloud Triton              │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│ Monitoring + Feedback Loop                    │
│ Prometheus/Grafana/Loki + active learning     │
└──────────────────────────────────────────────┘
```

---

# 1. Layer 1 — Device/Firmware

## Technology

```text
ESP-IDF C/C++
```

## Responsibilities

```text
CSI capture
node identity
binary packet encode
heartbeat
NVS config
WiFi reconnect
watchdog
OTA signed firmware
crash log
```

## Required packet fields

```text
magic
protocol_version
node_id
seq
timestamp_us
rssi
noise_floor
channel
n_subcarriers
I/Q or amp/phase payload
crc
firmware_version
```

## Hard rules

```text
No Arduino production firmware
No JSON raw CSI packets
No anonymous node
No packet without seq/timestamp
No OTA without signing/rollback
```

---

# 2. Layer 2 — Edge Gateway

## Technology

```text
Rust + Tokio
SQLite local buffer
ONNX Runtime optional
NATS client
OpenTelemetry
```

## Responsibilities

```text
receive UDP from 4 ESP32 nodes
validate packet magic/version/length/CRC
detect packet drop via seq
sync or normalize timestamps
local ring buffer for online inference
local raw cache for network outage
upload Bronze raw data
publish node health
run edge ONNX inference when enabled
manage model download/canary/rollback
```

## Services

```text
rf-gateway
rf-node-manager
rf-uploader
rf-inference-edge
```

## Gateway output contracts

```text
NATS topic: csi.raw.{deployment_id}.{node_id}
NATS topic: csi.batch.{deployment_id}
NATS topic: node.health.{deployment_id}
NATS topic: inference.{deployment_id}
Local buffer: SQLite + append-only packet files
```

---

# 3. Layer 3 — Streaming/Ingestion

## Technology

```text
NATS JetStream
FastAPI or Rust Axum
PostgreSQL
TimescaleDB
```

## Topics

```text
csi.raw.{deployment_id}.{node_id}
csi.batch.{deployment_id}
node.health.{deployment_id}
events.{deployment_id}
inference.{deployment_id}
```

## API

```text
POST /api/v1/deployments/{id}/csi/batch
POST /api/v1/deployments/{id}/events
POST /api/v1/deployments/{id}/health
GET  /api/v1/deployments/{id}/status
POST /api/v1/recording-sessions
POST /api/v1/training-jobs
GET  /api/v1/training-jobs/{id}
POST /api/v1/models/{version}/promote
POST /api/v1/models/{version}/rollback
```

---

# 4. Layer 4 — Data Lake

## Technology

```text
MinIO / S3
lakeFS optional
Parquet
Zarr / WebDataset
JSON metadata
```

## Canonical layout

```text
s3://rfpose/
  bronze/
    deployment=room01/
      date=2026-05-02/
        csi_raw/
        node_health/
        events/
        video/
        metadata.json

  silver/
    deployment=room01/
      session=001/
        csi_decoded.parquet
        pose_labels.parquet
        events.parquet
        quality_report.json

  gold/
    dataset_version=rfpose-room01-v0007/
      train/
      val/
      test/
      manifest.json
      stats.json
      normalization.json

  models/
    rfworldpose/
      version=0.4.12/
        model.pt
        model.onnx
        metrics.json
        model_card.md
```

## Principles

```text
Bronze immutable
Silver cleaned
Gold ML-ready
Helios $SCRATCH is cache only, never source of truth
All datasets and models are reproducible by version
```

---

# 5. Layer 5 — ETL/ELT Pipeline

## Technology

```text
Dagster
Python
Polars
PyArrow
NumPy/SciPy
Pandera or Great Expectations
```

## Asset graph

```text
raw_csi_sessions
  ↓
decoded_csi_frames
  ↓
baseline_profiles
  ↓
synced_multinode_csi
  ↓
teacher_pose_labels
  ↓
aligned_csi_pose
  ↓
csi_windows
  ↓
quality_reports
  ↓
gold_dataset
  ↓
dataset_registry_entry
```

## Quality gates

```text
node_count == 4
packet_drop_rate < threshold
fps_stability pass
RSSI range valid
timestamp monotonic
label_confidence > threshold
missing_windows < threshold
```

A session that fails quality gates may remain in Bronze/Silver but must not enter Gold unless explicitly marked as degraded/test data.

---

# 6. Layer 6 — Labeling / Teacher Pipeline

## Technology

```text
FFmpeg
OpenCV
MediaPipe / YOLO Pose / ViTPose
Detectron2 DensePose optional
SMPLer-X optional
```

## Flow

```text
video.mp4
→ frame extraction
→ pose teacher model
→ pose_labels.jsonl/parquet
→ confidence filtering
→ align with CSI timestamp
```

## Teacher versions

```text
teacher_pose=vitpose-v1
teacher_densepose=detectron2-densepose-v1
teacher_smpl=smplerx-v1
```

## Privacy rule

Camera data is train-time only. Production inference is WiFi-only. Video retention must be shorter than derived pose-label retention.

---

# 7. Layer 7 — Dataset Registry

## Technology

```text
PostgreSQL
MLflow tags optional
lakeFS/DVC optional
```

## Dataset version record

```text
dataset_version
source_sessions
preprocess_version
teacher_version
quality_thresholds
train_val_test_split
stats
artifact_uri
created_at
created_by
```

Example:

```text
rfpose-room01-v0007
```

---

# 8. Layer 8 — Helios GH200 Training Backend

## Role

```text
Batch training backend via Slurm
```

Helios does not host production API, dashboard, NATS, MinIO, or online inference.

## Helios details

```text
login: login01.helios.cyfronet.pl
partition: plgrid-gpu-gh200
node: 4x NVIDIA GH200 96GB
CPU: Grace ARM aarch64
OS: Rocky Linux 9
time limit: 48h
scheduler: Slurm
```

## Integration

```text
Dagster/Control Plane
→ Helios Slurm Submitter
→ sbatch
→ GH200 job pulls dataset from S3/MinIO
→ train/eval/export
→ upload artifacts
→ MLflow registry
```

## Slurm submitter module

```text
helios_runner/
  submit.py
  status.py
  cancel.py
  templates/train_gh200.sbatch
  sync.py
```

## Job lifecycle

```text
created
submitted
queued
running
uploading
completed
failed
cancelled
```

## Training job table

```text
id
dataset_version
train_config
backend='helios-slurm'
slurm_job_id
slurm_partition='plgrid-gpu-gh200'
status
artifact_uri
eval_report_uri
logs_uri
created_at
finished_at
```

## HPC rules

```text
Build/runtime must respect aarch64 GH200 compute architecture
Use module stack or Apptainer compatible with Helios
Checkpoint regularly; 48h time limit requires resume support
$SCRATCH is temporary cache only
All artifacts must be uploaded back to S3/MLflow
```

---

# 9. Layer 9 — Training/MLOps

## Technology

```text
PyTorch
Hydra
MLflow
torchrun/DDP
bf16 mixed precision
ONNX export
TensorRT optional
```

## Training stages

```text
1. self-supervised CSI pretrain
2. public dataset supervised train
3. local teacher-student train
4. room LoRA adapter fine-tune
5. eval
6. export ONNX
```

## Model architecture target

```text
RF-WorldPose:
CSI Tokenizer
+ RF Graph Transformer
+ Neural RF Field
+ SMPL/Skeleton Decoder
+ DensePose Head optional
+ Room LoRA Adapter
```

## Job types

```text
pretrain_ssl
supervised_train
room_lora_finetune
teacher_label_generation
model_eval
onnx_export
synthetic_generation
hard_case_mining
```

---

# 10. Layer 10 — Model Registry

## Technology

```text
MLflow Model Registry
MinIO/S3 artifacts
PostgreSQL metadata
```

## Model states

```text
candidate
staging
production
archived
rollback
```

## Artifact package

```text
model.pt
model.onnx
normalization.json
preprocess_config.yaml
model_config.yaml
metrics.json
eval_report.json
model_card.md
hash/signature
```

## Eval gates

```text
presence_f1 >= threshold
action_acc >= threshold
keypoint_error <= threshold
empty_false_positive <= threshold
temporal_jitter <= threshold
latency_p95 <= threshold
node_dropout_test pass
```

A model that fails eval gates cannot be promoted.

---

# 11. Layer 11 — Inference Deployment

## Edge mode

```text
Gateway + ONNX Runtime
```

Used for:

```text
low latency
privacy
offline operation
```

## Cloud mode

```text
NVIDIA Triton + TensorRT
```

Used for:

```text
large model inference
batch analysis
research/eval
```

## Realtime pipeline

```text
CSI stream
→ online preprocess
→ ring buffer 3s
→ inference every 100-200ms
→ temporal smoother
→ confidence gate
→ WebSocket/API output
```

## Output contract

```json
{
  "deployment_id": "room01",
  "model_version": "rfworldpose-v0.4.12",
  "timestamp": 1714663800.123,
  "presence": true,
  "action": "walking",
  "confidence": 0.82,
  "keypoints": [],
  "quality": {
    "nodes_online": 4,
    "packet_drop_rate": 0.01,
    "drift_score": 0.08
  }
}
```

---

# 12. Layer 12 — Dashboard

## Technology

```text
Next.js
WebSocket
Three.js / Canvas
ECharts/uPlot
```

## Screens

```text
Live skeleton
Node health
CSI raw graph
Recording control
Dataset sessions
Training jobs
Model registry
Alerts
```

---

# 13. Layer 13 — Monitoring/Observability

## Technology

```text
Prometheus
Grafana
Loki
OpenTelemetry
Alertmanager
DCGM Exporter for GH200
```

## Monitor

```text
ESP32 node health
gateway packet rate
drop rate
CSI drift
model confidence
inference latency
training job status
GH200 GPU utilization
storage usage
API errors
```

## Alerts

```text
node offline > 60s
packet drop > 10%
inference latency p95 > 200ms
drift score high
disk free < 15%
GH200 job failed
model confidence collapse
```

---

# 14. Layer 14 — Security

## Device

```text
device identity key
signed firmware
OTA rollback
secure config
```

## Transport

```text
TLS/mTLS gateway-cloud
signed packets or gateway auth
```

## Data

```text
S3 encryption
RBAC
audit log
video retention policy
```

## Secrets

```text
SOPS + age
or Vault later
```

---

# Final deployment standard

There is one final standard, not separate v1/v2 architecture. The system may be implemented in milestones, but all code and docs target the final architecture.

## Required runtime stack

```text
Firmware:    ESP-IDF C/C++
Gateway:     Rust/Tokio
Transport:   UDP local, NATS JetStream upstream
API:         FastAPI or Rust Axum
DB:          PostgreSQL + TimescaleDB
Data Lake:   MinIO/S3
ETL:         Dagster + Polars/PyArrow
Quality:     Pandera/Great Expectations
Training:    PyTorch + Hydra + MLflow
HPC:         Helios Slurm plgrid-gpu-gh200
Serving:     ONNX Runtime edge, Triton/TensorRT cloud
Dashboard:   Next.js + WebSocket + Three.js
Monitoring:  Prometheus + Grafana + Loki + OpenTelemetry
Security:    mTLS, signed firmware, SOPS/Vault
Deploy:      Docker Compose for lab/single-site, k3s/K8s for managed multi-service deployment
```

---

# Production completion criteria

The platform is production-complete when all items below are true:

```text
✅ 4 ESP32-S3 stream CSI ổn định
✅ Gateway validate + buffer + upload
✅ Data lake Bronze/Silver/Gold
✅ Dataset versioning
✅ Teacher label pipeline
✅ ETL quality checks
✅ Helios Slurm submitter
✅ GH200 train/eval/export tự động
✅ MLflow model registry
✅ Eval gate trước deploy
✅ Edge ONNX inference
✅ Dashboard live
✅ Monitoring/alerts
✅ OTA firmware
✅ Rollback model
✅ Feedback hard-case loop
```

---

# Production flows

## Data collection flow

```text
1. Operator tạo recording session
2. Gateway kiểm tra đủ 4 node online
3. Thu empty baseline
4. Thu scenario/action
5. Upload Bronze raw
6. Dagster ETL tạo Silver
7. Teacher pipeline tạo pose labels
8. Align CSI + pose
9. Quality check
10. Gold dataset version created
```

## Helios training flow

```text
1. Dataset version ready
2. Dagster tạo training_job
3. Helios submitter render sbatch
4. SSH login01.helios.cyfronet.pl
5. sbatch vào plgrid-gpu-gh200
6. Job pull dataset từ MinIO/S3
7. torchrun train trên 4 GH200
8. Eval + export ONNX
9. Upload artifact về S3/MLflow
10. Register model candidate
11. Eval gate
12. Promote staging/production
```

## Deployment flow

```text
1. Model production version mới approved
2. Gateway nhận update event
3. Download model.onnx + normalization
4. Verify hash/signature
5. Warmup model
6. Canary switch
7. Monitor confidence/latency
8. Promote full
9. Rollback nếu fail
```

## Feedback loop

```text
1. Production phát hiện uncertainty/drift
2. Gateway lưu hard windows
3. Upload hard cases
4. Dagster tạo hard_case_dataset
5. Helios fine-tune adapter
6. Eval against golden set
7. Deploy improved model
```

---

# Final assessment

This architecture is strong because it treats RF sensing as a full data/ML product, not a notebook model. It has the required production primitives: immutable raw data, quality gates, dataset lineage, managed HPC training, model registry, eval gates, edge/cloud serving, rollback, observability, security, and active learning.

The riskiest areas are:

1. CSI packet fidelity and synchronization across four ESP32-S3 nodes.
2. Local dataset quality and label alignment with camera teacher.
3. ARM/aarch64 dependency compatibility on Helios GH200.
4. Domain adaptation between public/pretrained data and the target room.
5. Avoiding training-serving skew between Dagster preprocessing and edge online preprocessing.

Mitigation is already built into the architecture: shared preprocessing library, dataset quality gates, Helios job isolation, MLflow registry, eval gates, and hard-case feedback loop.
