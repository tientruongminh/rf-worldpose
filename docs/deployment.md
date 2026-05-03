# Deployment Guide

## Deployment targets
1. Single-site/lab: Docker Compose + systemd gateway.
2. Managed production: k3s/Kubernetes + gateway DaemonSet + external S3/Postgres/NATS.

## Required services
```text
PostgreSQL + TimescaleDB optional
NATS JetStream
MinIO/S3
MLflow
Dagster
API service
Dashboard
Prometheus
Grafana
Loki
```

## Gateway systemd unit
```ini
[Unit]
Description=RF-WorldPose Gateway
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/rf-worldpose/gateway/rf-gateway
Environment=RFPOSE_DEPLOYMENT_ID=room01
Environment=RFPOSE_GATEWAY_BIND=0.0.0.0:5006
Environment=RFPOSE_GATEWAY_SQLITE=/var/lib/rfpose/gateway.sqlite
Environment=NATS_URL=nats://control-plane:4222
Environment=S3_BUCKET=rfpose
Environment=S3_ENDPOINT_URL=https://s3.example.com
ExecStart=/usr/local/bin/rf-gateway
Restart=always
RestartSec=5
User=rfpose

[Install]
WantedBy=multi-user.target
```

## Kubernetes/k3s
Manifests start in `infra/k8s/base/`. Production should add Ingress/TLS, SOPS/Vault secrets, persistent volumes, requests/limits, NetworkPolicies, and PodDisruptionBudgets.

## Triton serving
Place ONNX at `infra/triton/model_repository/rfworldpose/1/model.onnx` and run:
```bash
tritonserver --model-repository infra/triton/model_repository
```
TensorRT engines must be built on the target GPU class.
