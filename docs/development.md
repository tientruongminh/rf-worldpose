# Development Guide

## Bootstrap

```bash
cp .env.example .env
docker compose -f infra/docker-compose/docker-compose.yml up -d
```

## Milestones

1. Firmware packet schema + gateway decoder.
2. Bronze ingest + local buffer.
3. Dagster Bronze→Silver→Gold ETL.
4. Teacher label pipeline.
5. Helios Slurm submitter.
6. MLflow model registry + eval gates.
7. Edge ONNX inference + dashboard.
8. Monitoring, OTA, rollback, feedback loop.
