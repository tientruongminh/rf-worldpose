# RF-WorldPose Platform

Production/research platform for WiFi CSI human sensing and WiFi-only skeleton/DensePose inference.

## Final architecture

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

The canonical architecture spec is [`docs/final-architecture.md`](docs/final-architecture.md).

## Mission

```text
Thu CSI từ ESP32-S3
→ build dataset chuẩn
→ train model trên Helios GH200
→ deploy WiFi-only skeleton/DensePose inference
→ monitor + feedback + cải thiện liên tục
```

## Production completion criteria

- [ ] 4 ESP32-S3 stream CSI ổn định
- [ ] Gateway validate + buffer + upload
- [ ] Data lake Bronze/Silver/Gold
- [ ] Dataset versioning
- [ ] Teacher label pipeline
- [ ] ETL quality checks
- [ ] Helios Slurm submitter
- [ ] GH200 train/eval/export tự động
- [ ] MLflow model registry
- [ ] Eval gate trước deploy
- [ ] Edge ONNX inference
- [ ] Dashboard live
- [ ] Monitoring/alerts
- [ ] OTA firmware
- [ ] Rollback model
- [ ] Feedback hard-case loop

## Quick start

```bash
cp .env.example .env
docker compose -f infra/docker-compose/docker-compose.yml up -d
```

## Services

- API: http://localhost:8080
- Dashboard: http://localhost:3000
- MinIO: http://localhost:9001
- MLflow: http://localhost:5000
- Dagster: http://localhost:3001
- Grafana: http://localhost:3002
- Prometheus: http://localhost:9090

## Repository layout

```text
firmware/esp32-csi-node/      ESP-IDF firmware
 gateway/rf-gateway/          Rust/Tokio edge gateway
 services/                    API, ingest, inference, model registry
 pipelines/dagster/           ETL Bronze/Silver/Gold
 helios_runner/               Helios Slurm submitter
 ml/                          PyTorch training/eval/export
 libs/                        shared schemas/preprocess/core
 dashboard/                   Next.js + WebSocket + Three.js
 infra/                       compose/k8s/monitoring/security
 docs/                        final architecture and implementation docs
```
