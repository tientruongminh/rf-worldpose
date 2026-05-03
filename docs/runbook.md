# RF-WorldPose Production Runbook

## Purpose
This runbook describes how to operate RF-WorldPose in production/research environments: bring-up, data collection, ETL, training on Helios GH200, deployment, monitoring, rollback, and incident handling.

## Golden path
```text
1. Bring up control-plane infrastructure
2. Register deployment + nodes
3. Start Rust gateway
4. Flash/provision ESP32-S3 nodes
5. Record CSI session
6. Upload Bronze data
7. Run Bronze→Silver→Gold ETL
8. Register dataset version
9. Submit Helios training job
10. Evaluate and package model
11. Promote model
12. Gateway downloads/runs edge model
13. Monitor drift/failures
14. Feed hard cases back into training
```

## Local stack bring-up
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

## Gateway smoke test
```bash
cd gateway/rf-gateway
RFPOSE_DEPLOYMENT_ID=room01 RFPOSE_GATEWAY_BIND=0.0.0.0:5006 RFPOSE_GATEWAY_SQLITE=/tmp/rfpose-gateway.sqlite cargo run --release
python tools/mock_sender/send_mock_csi.py --node-id 1 --count 100
```

## Incident checklist
### Node offline
1. Check gateway logs.
2. Check node power/USB.
3. Check WiFi RSSI/channel.
4. Confirm UDP packets with tcpdump.
5. Reboot one node, not all nodes at once.

### Packet drop high
1. Check node RSSI.
2. Reduce CSI sample rate.
3. Check gateway CPU/disk.
4. Check WiFi interference/channel.
5. Inspect SQLite buffer growth.

### Bronze upload failing
1. Check MinIO/S3 credentials.
2. Check bucket exists.
3. Check gateway network.
4. Confirm local SQLite still buffering.
5. Restart only after confirming buffer file is safe.

### Model confidence collapse
1. Switch to degraded state.
2. Roll back previous model.
3. Capture hard-case windows.
4. Check room layout changed.
5. Run adapter fine-tune.
