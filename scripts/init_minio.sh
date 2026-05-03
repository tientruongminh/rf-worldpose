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
