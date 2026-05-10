#!/usr/bin/env bash
# ==========================================================
# RF-WorldPose — Khởi động local dev stack (1 lệnh duy nhất)
# ==========================================================
# Chạy:  ./scripts/dev_up.sh
# Dừng:  make down   (hoặc docker compose ... down)
# Reset: docker compose -f infra/docker-compose/docker-compose.yml down -v
# ==========================================================
set -euo pipefail
cd "$(dirname "$0")/.."

cp -n .env.example .env 2>/dev/null || true

COMPOSE="docker compose -f infra/docker-compose/docker-compose.yml --env-file .env"

echo "==> Starting all services..."
$COMPOSE up -d

echo "==> Waiting for PostgreSQL..."
until $COMPOSE exec -T postgres pg_isready -U rfpose >/dev/null 2>&1; do
  sleep 1
done

echo "==> Running database migrations..."
export DATABASE_URL="postgresql://rfpose:rfpose@localhost:5432/rfpose"
for f in infra/postgres/migrations/*.sql; do
  echo "    Applying $f"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f "$f" 2>/dev/null || true
done

echo ""
echo "=========================================="
echo "  RF-WorldPose stack is ready!"
echo "=========================================="
echo ""
echo "  API (Swagger)    http://localhost:8080/docs"
echo "  MLflow           http://localhost:5000"
echo "  Dagster          http://localhost:3004"
echo "  MinIO Console    http://localhost:9003"
echo "                   user: rfpose / pass: rfpose-secret"
echo "  Grafana          http://localhost:3002"
echo "                   user: admin / pass: admin"
echo "  Prometheus       http://localhost:9090"
echo "  NATS Monitor     http://localhost:8222"
echo "  PostgreSQL       localhost:5432"
echo "                   user: rfpose / pass: rfpose"
echo ""
