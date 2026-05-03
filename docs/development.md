# Development Guide

## Bootstrap

```bash
cp .env.example .env
docker compose -f infra/docker-compose/docker-compose.yml up -d
```

## Build order

The build order is documented in [`implementation-plan.md`](implementation-plan.md). The project has one final architecture; milestones are only execution order, not separate product versions.

## Local services

- Postgres: metadata/control plane
- NATS JetStream: event/stream transport
- MinIO: local S3-compatible data lake
- MLflow: experiment/model registry
- Dagster: ETL orchestration
- Prometheus/Grafana/Loki: observability
