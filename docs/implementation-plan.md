# RF-WorldPose Implementation Plan

This is not a reduced architecture. It is the build order for the final architecture.

## Milestone A — Contracts and control-plane foundation

- Protobuf CSI/health/event schemas.
- Postgres schema for deployments, nodes, sessions, dataset versions, model versions, training jobs, eval gates.
- API endpoints for deployments, sessions, events, training jobs, model promotion/rollback.
- Local Docker Compose stack: Postgres, NATS, MinIO, MLflow, Dagster, Prometheus, Grafana, Loki.

## Milestone B — Device and gateway path

- ESP-IDF firmware packet encoder.
- Rust gateway CSI decoder/validator.
- Local buffer and node health tracking.
- NATS publisher and Bronze uploader.
- Gateway metrics via OpenTelemetry/Prometheus.

## Milestone C — Data lake and ETL

- Bronze raw ingestion layout.
- Silver decoder and quality report.
- Teacher label pipeline.
- CSI/video timestamp alignment.
- Gold windowed dataset builder.
- Dataset registry entry creation.

## Milestone D — Helios training backend

- Slurm submitter render/upload/sbatch/status/cancel.
- GH200 sbatch template with checkpoint/resume.
- Training/eval/export jobs.
- MLflow logging and artifact upload.
- Eval gate automation.

## Milestone E — Serving and dashboard

- Edge ONNX inference path.
- Cloud Triton/TensorRT path.
- WebSocket output stream.
- Next.js dashboard: live skeleton, node health, dataset sessions, training jobs, model registry, alerts.
- Canary deploy and rollback.

## Milestone F — Security and operations

- mTLS gateway-cloud.
- Device identity.
- Signed firmware and OTA rollback.
- SOPS/Vault secrets.
- Alerts for node failure, drop rate, drift, inference latency, GH200 job failure.
- Hard-case feedback loop.
