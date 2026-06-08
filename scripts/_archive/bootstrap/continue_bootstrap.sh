#!/usr/bin/env bash
set -euo pipefail

mkdir -p scripts tools/mock_sender services/api/scripts pipelines/dagster/rfpose_pipelines/resources
cat > scripts/init_minio.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
: "${MINIO_ALIAS:=local}"
: "${MINIO_ENDPOINT:=http://localhost:9000}"
: "${MINIO_ROOT_USER:=rfpose}"
: "${MINIO_ROOT_PASSWORD:=rfpose-secret}"
: "${S3_BUCKET:=rfpose}"
if ! command -v mc >/dev/null 2>&1; then echo "mc (MinIO client) is required" >&2; exit 1; fi
mc alias set "$MINIO_ALIAS" "$MINIO_ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mb --ignore-existing "$MINIO_ALIAS/$S3_BUCKET"
for prefix in bronze silver gold models mlflow; do echo '{}' | mc pipe "$MINIO_ALIAS/$S3_BUCKET/$prefix/.keep" >/dev/null; done
echo "MinIO bucket initialized: $S3_BUCKET"
EOS
chmod +x scripts/init_minio.sh

cat > scripts/run_migrations.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
: "${DATABASE_URL:=postgresql://rfpose:rfpose@localhost:5432/rfpose}"
for f in infra/postgres/migrations/*.sql; do echo "Applying $f"; psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$f"; done
EOS
chmod +x scripts/run_migrations.sh

cat > scripts/dev_up.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
cp -n .env.example .env || true
docker compose -f infra/docker-compose/docker-compose.yml up -d
echo "Waiting for Postgres..."
until docker compose -f infra/docker-compose/docker-compose.yml exec -T postgres pg_isready -U rfpose >/dev/null 2>&1; do sleep 1; done
echo "Services are up. Run scripts/run_migrations.sh and scripts/init_minio.sh if tools are installed on host."
EOS
chmod +x scripts/dev_up.sh

cat > tools/mock_sender/send_mock_csi.py <<'EOS'
#!/usr/bin/env python3
from __future__ import annotations
import argparse, socket, struct, time, zlib, math
MAGIC = 0xC5110001
HEADER_LEN = 32
PROTO = 1

def build_packet(node_id:int, seq:int, n_sub:int=56, channel:int=6, rssi:int=-50, fw:int=1) -> bytes:
    timestamp_us = int(time.time() * 1_000_000)
    iq=[]
    for i in range(n_sub):
        iq.extend([int(20*math.sin(i/5)), int(20*math.cos(i/7))])
    payload = b''.join(struct.pack('<h', x) for x in iq)
    header = struct.pack('<IBBHQbbBBHHI', MAGIC, PROTO, node_id, HEADER_LEN, seq, timestamp_us, rssi, -90, channel, 0, n_sub, fw, len(payload))
    crc = zlib.crc32(header + payload) & 0xffffffff
    return header + payload + struct.pack('<I', crc)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5006)
    ap.add_argument('--node-id', type=int, default=1)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--hz', type=float, default=20)
    args=ap.parse_args()
    sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    delay=1.0/args.hz
    for seq in range(args.count):
        sock.sendto(build_packet(args.node_id, seq), (args.host,args.port))
        time.sleep(delay)
    print(f"sent {args.count} packets to {args.host}:{args.port}")
if __name__=='__main__': main()
EOS
chmod +x tools/mock_sender/send_mock_csi.py

cat > services/api/Dockerfile <<'EOS'
FROM python:3.11-slim
WORKDIR /app
COPY services/api /app/services/api
RUN pip install --no-cache-dir -e /app/services/api
CMD ["uvicorn", "rfpose_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
EOS

cat > infra/docker-compose/docker-compose.yml <<'EOS'
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-rfpose}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-rfpose}
      POSTGRES_DB: ${POSTGRES_DB:-rfpose}
    ports: ["5432:5432"]
    volumes:
      - postgres:/var/lib/postgresql/data
      - ../postgres/migrations:/docker-entrypoint-initdb.d:ro
  nats:
    image: nats:2.10
    command: ["-js", "-m", "8222"]
    ports: ["4222:4222", "8222:8222"]
  minio:
    image: minio/minio:latest
    command: server /data --console-address ':9001'
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER:-rfpose}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-rfpose-secret}
    ports: ["9000:9000", "9001:9001"]
    volumes: ["minio:/data"]
  api:
    image: python:3.11-slim
    working_dir: /app
    command: sh -c "pip install -e services/api && uvicorn rfpose_api.main:app --host 0.0.0.0 --port 8080"
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-rfpose}:${POSTGRES_PASSWORD:-rfpose}@postgres:5432/${POSTGRES_DB:-rfpose}
      HELIOS_LOGIN: ${HELIOS_LOGIN:-login01.helios.cyfronet.pl}
      HELIOS_ACCOUNT: ${HELIOS_ACCOUNT:-CHANGE_ME-gpu-gh200}
      HELIOS_PARTITION: ${HELIOS_PARTITION:-plgrid-gpu-gh200}
      S3_BUCKET: ${S3_BUCKET:-rfpose}
      S3_ENDPOINT_URL: http://minio:9000
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports: ["8080:8080"]
    volumes: ["../../:/app"]
    depends_on: [postgres]
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root s3://rfpose/mlflow
    environment:
      AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER:-rfpose}
      AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD:-rfpose-secret}
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    ports: ["5000:5000"]
    volumes: ["mlflow:/mlflow"]
    depends_on: [minio]
  dagster:
    image: python:3.11-slim
    working_dir: /app
    command: sh -c "pip install -e pipelines/dagster && dagster dev -h 0.0.0.0 -p 3001 -m rfpose_pipelines"
    environment:
      DAGSTER_HOME: /app/.dagster
      S3_ENDPOINT_URL: http://minio:9000
      S3_BUCKET: ${S3_BUCKET:-rfpose}
      DATABASE_URL: postgresql://${POSTGRES_USER:-rfpose}:${POSTGRES_PASSWORD:-rfpose}@postgres:5432/${POSTGRES_DB:-rfpose}
    ports: ["3001:3001"]
    volumes: ["../../:/app"]
    depends_on: [postgres, minio]
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes: ["../monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro"]
  grafana:
    image: grafana/grafana:latest
    ports: ["3002:3000"]
    volumes: ["grafana:/var/lib/grafana"]
  loki:
    image: grafana/loki:2.9.8
    command: -config.file=/etc/loki/local-config.yaml
    ports: ["3100:3100"]
volumes:
  postgres:
  minio:
  mlflow:
  grafana:
EOS

cat > infra/monitoring/prometheus/prometheus.yml <<'EOS'
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: prometheus
    static_configs:
      - targets: ['localhost:9090']
  - job_name: rfpose-api
    metrics_path: /health
    static_configs:
      - targets: ['api:8080']
  - job_name: nats
    static_configs:
      - targets: ['nats:8222']
EOS

cat >> README.md <<'EOS'

## Self-install infrastructure later

When ready to install/run the local stack:

```bash
./scripts/dev_up.sh
export DATABASE_URL=postgresql://rfpose:rfpose@localhost:5432/rfpose
./scripts/run_migrations.sh
export MINIO_ENDPOINT=http://localhost:9000
export MINIO_ROOT_USER=rfpose
export MINIO_ROOT_PASSWORD=rfpose-secret
export S3_BUCKET=rfpose
./scripts/init_minio.sh
```

Test gateway with mock packets:

```bash
cd gateway/rf-gateway
RFPOSE_DEPLOYMENT_ID=room01 \
RFPOSE_GATEWAY_SQLITE=/tmp/rfpose-gateway.sqlite \
NATS_URL=nats://localhost:4222 \
S3_BUCKET=rfpose \
S3_ENDPOINT_URL=http://localhost:9000 \
AWS_ACCESS_KEY_ID=rfpose \
AWS_SECRET_ACCESS_KEY=rfpose-secret \
cargo run

# in another terminal, from repo root
python tools/mock_sender/send_mock_csi.py --node-id 1 --count 100
```
EOS

python3 -m compileall services/api/src tools/mock_sender pipelines/dagster/rfpose_pipelines >/tmp/rfpose_py_compile2.log
cargo test --manifest-path gateway/rf-gateway/Cargo.toml --quiet

git add .
git commit -m "chore: add infrastructure bootstrap scripts and mock E2E sender"
git status --short
git log --oneline -8
