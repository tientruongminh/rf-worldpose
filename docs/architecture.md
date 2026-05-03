# RF-WorldPose Architecture

## Final design

ESP32-S3 CSI mesh → Rust Edge Gateway → NATS/Ingest → MinIO/S3 Data Lake → Dagster ETL Bronze/Silver/Gold → Dataset Registry → Helios GH200 Slurm Training → MLflow Model Registry → ONNX/Triton Deployment → Monitoring + OTA + Feedback Loop.

## Planes

- **Data plane:** firmware, gateway, ingest, object storage.
- **ML plane:** ETL, dataset registry, training, evaluation, model registry.
- **Serving plane:** edge ONNX Runtime, cloud Triton/TensorRT.
- **Ops plane:** monitoring, security, deployment, OTA, rollback.

## Helios integration

Helios is a Slurm batch training backend, not a production service host. Jobs target `plgrid-gpu-gh200` and use GH200 aarch64 compute nodes for train/eval/export. The source of truth remains MinIO/S3 + MLflow outside Helios.
