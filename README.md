<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&height=220&color=0:020617,55:064e3b,100:111827&text=RF-WorldPose&fontColor=f8fafc&fontSize=54&fontAlignY=38&desc=WiFi%20CSI%20sensing%20platform%20for%20RF%20human%20perception&descAlignY=58&descSize=16">
    <img alt="RF-WorldPose" src="https://capsule-render.vercel.app/api?type=waving&height=220&color=0:f8fafc,55:d1fae5,100:e5e7eb&text=RF-WorldPose&fontColor=111827&fontSize=54&fontAlignY=38&desc=WiFi%20CSI%20sensing%20platform%20for%20RF%20human%20perception&descAlignY=58&descSize=16">
  </picture>
</p>

<p align="center">
  <a href="docs/final-architecture.md"><img alt="Architecture" src="https://img.shields.io/badge/architecture-production--oriented-064e3b?style=for-the-badge"></a>
  <a href="docs/helios.md"><img alt="Helios" src="https://img.shields.io/badge/HPC-Helios%20GH200-0f172a?style=for-the-badge"></a>
  <a href="docs/mlops.md"><img alt="MLOps" src="https://img.shields.io/badge/MLOps-Bronze%20%7C%20Silver%20%7C%20Gold-14532d?style=for-the-badge"></a>
  <a href="docs/security.md"><img alt="Security" src="https://img.shields.io/badge/security-mTLS%20%7C%20signed%20OTA-111827?style=for-the-badge"></a>
</p>

<p align="center">
  <b>RF-WorldPose</b> is a production/research platform for WiFi CSI human sensing: four ESP32-S3 nodes capture RF channel state, a Rust gateway turns packets into reliable data streams, and a full MLOps pipeline trains WiFi-only skeleton/DensePose models on Helios GH200.
</p>

---

## Why this exists

Human perception normally depends on cameras. RF-WorldPose explores a harder path: infer presence, motion, skeleton structure, and DensePose-style representations from WiFi CSI alone.

The repository is built as a real platform rather than a notebook demo: firmware, edge gateway, control plane, data lake, ETL, HPC training, model governance, serving, monitoring, and security are separated into production-style contracts.

```text
ESP32-S3 CSI mesh
   -> Rust/Tokio edge gateway
      -> NATS JetStream + local buffer
         -> MinIO/S3 Bronze lake
            -> Dagster ETL: Bronze -> Silver -> Gold
               -> Dataset registry
                  -> Helios GH200 Slurm training
                     -> MLflow + model card + eval gates
                        -> ONNX edge serving / Triton cloud serving
                           -> monitoring, rollback, feedback loop
```

## System at a glance

| Layer | What it does | Implementation |
| --- | --- | --- |
| Sensor firmware | Captures WiFi CSI, encodes CRC-protected packets, streams UDP | ESP-IDF C/C++ |
| Edge gateway | Validates packets, tracks drops, buffers locally, publishes upstream | Rust, Tokio, SQLite, NATS |
| Control plane | Deployments, nodes, sessions, datasets, training jobs, model registry | FastAPI, PostgreSQL |
| Data lake | Immutable Bronze, decoded Silver, ML-ready Gold datasets | MinIO/S3, Dagster, Polars, PyArrow |
| Training | RFWorldPose model, LoRA adapters, distillation, evaluation | PyTorch, Hydra, MLflow |
| HPC backend | Batch training and artifact export on GH200 nodes | Helios Slurm `plgrid-gpu-gh200` |
| Serving | Edge inference and cloud inference contracts | ONNX Runtime, Triton/TensorRT |
| Operations | Metrics, logs, dashboards, deployment manifests, security posture | Prometheus, Grafana, Loki, k8s, mTLS |

## Repository layout

```text
firmware/esp32-csi-node/      ESP32-S3 CSI firmware, packet encoder, provisioning
 gateway/rf-gateway/          Rust/Tokio gateway: UDP, CRC, buffer, NATS, S3, metrics
 services/api/                FastAPI control plane and registry endpoints
 pipelines/dagster/           Bronze -> Silver -> Gold ETL assets and transforms
 helios_runner/               Slurm template rendering, submit, status, cancel tools
 ml/rfpose/                   PyTorch models, training, eval, export, packaging
 dashboard/                   Next.js operations UI and product landing page
 infra/                       Docker Compose, k8s, Triton, monitoring, security
 docs/                        Architecture, runbooks, deployment, security, MLOps
 tools/mock_sender/           Synthetic CSI packet sender for gateway smoke tests
```

## What is implemented

- ESP32 CSI packet contract with native C unit test
- ESP-IDF CSI callback path and UDP streamer
- Rust gateway packet decoder, CRC validation, local SQLite buffer, NATS hooks, S3 Bronze uploader
- FastAPI control plane for deployments, sessions, datasets, training jobs, and models
- PostgreSQL schema for platform metadata
- Dagster/Polars ETL from Bronze to Silver and Gold
- Dataset registration helper
- RFWorldPose model, training, evaluation, export, and eval gates
- LoRA adapter and knowledge distillation training path
- Model artifact packager with model card and SHA256 manifest
- Helios GH200 Slurm submitter with dry-run test
- Triton model repository contract and ONNX serving path
- Docker Compose lab stack and Kubernetes base manifests
- Prometheus/Grafana/Loki monitoring scaffold
- mTLS, SOPS/Vault, signed OTA security docs
- Production runbook and deployment documentation

## Quick start

Bring up the local control-plane stack:

```bash
cp .env.example .env
./scripts/dev_up.sh
export DATABASE_URL=postgresql://rfpose:rfpose@localhost:5432/rfpose
./scripts/run_migrations.sh
export MINIO_ENDPOINT=http://localhost:9000
export MINIO_ROOT_USER=rfpose
export MINIO_ROOT_PASSWORD=rfpose-secret
export S3_BUCKET=rfpose
./scripts/init_minio.sh
```

Run the gateway and send synthetic CSI:

```bash
cd gateway/rf-gateway
RFPOSE_DEPLOYMENT_ID=room01 \
RFPOSE_GATEWAY_BIND=0.0.0.0:5006 \
RFPOSE_GATEWAY_SQLITE=/tmp/rfpose-gateway.sqlite \
NATS_URL=nats://localhost:4222 \
S3_BUCKET=rfpose \
S3_ENDPOINT_URL=http://localhost:9000 \
AWS_ACCESS_KEY_ID=rfpose \
AWS_SECRET_ACCESS_KEY=rfpose-secret \
cargo run
```

```bash
python tools/mock_sender/send_mock_csi.py --node-id 1 --count 100
```

## Validation

```bash
make -C firmware/esp32-csi-node/test check
cargo test --manifest-path gateway/rf-gateway/Cargo.toml
python -m compileall services/api/src pipelines/dagster/rfpose_pipelines ml/rfpose
PYTHONPATH=helios_runner python helios_runner/test_dry_run.py
npm --prefix dashboard run build
```

## Documentation

Start with the documentation index:

- [`docs/index.md`](docs/index.md)
- [`docs/final-architecture.md`](docs/final-architecture.md)
- [`docs/runbook.md`](docs/runbook.md)
- [`docs/deployment.md`](docs/deployment.md)
- [`docs/security.md`](docs/security.md)
- [`docs/mlops.md`](docs/mlops.md)
- [`docs/helios.md`](docs/helios.md)

## Production readiness status

RF-WorldPose is currently a **production-oriented research codebase**. It has the architecture, contracts, scaffolds, and smoke-test paths required for a serious bring-up.

It is not yet a production-proven deployment until the following are validated on real infrastructure:

- four ESP32-S3 nodes streaming CSI continuously
- long-running gateway ingestion and Bronze upload
- real MinIO/Postgres/NATS/Dagster deployment
- real Helios GH200 Slurm jobs with artifact export
- ONNX/Triton inference benchmarks
- mTLS certificates and signed OTA rollout
- 24-hour soak tests and failure recovery drills

## License

Research prototype. Add a formal license before public distribution.
