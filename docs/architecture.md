# Architecture

The canonical architecture document is [`final-architecture.md`](final-architecture.md).

Short form:

```text
ESP32-S3 CSI mesh
→ Rust Edge Gateway
→ NATS/Ingest
→ MinIO/S3 Data Lake
→ Dagster ETL Bronze/Silver/Gold
→ Dataset Registry
→ Helios GH200 Slurm Training
→ MLflow Model Registry
→ ONNX/Triton Deployment
→ Monitoring + OTA + Feedback Loop
```

There is one final target architecture. Implementation may be staged, but architecture, schemas, APIs, and repo layout all target the final platform.
