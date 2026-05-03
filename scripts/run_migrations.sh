#!/usr/bin/env bash
set -euo pipefail
: "${DATABASE_URL:=postgresql://rfpose:rfpose@localhost:5432/rfpose}"
for f in infra/postgres/migrations/*.sql; do echo "Applying $f"; psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$f"; done
