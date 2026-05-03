#!/usr/bin/env bash
set -euo pipefail
cp -n .env.example .env || true
docker compose -f infra/docker-compose/docker-compose.yml up -d
echo "Waiting for Postgres..."
until docker compose -f infra/docker-compose/docker-compose.yml exec -T postgres pg_isready -U rfpose >/dev/null 2>&1; do sleep 1; done
echo "Services are up. Run scripts/run_migrations.sh and scripts/init_minio.sh if tools are installed on host."
