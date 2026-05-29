#!/usr/bin/env python3
"""Deploy Job Portal on VPS: clone repo, install deps, setup DB, start API."""
import sys, io, paramiko, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST = "207.180.243.242"
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username="root", password="teamKDL123456", timeout=15)
print("Connected to VPS.\n")


def run(cmd, timeout=120):
    print(f"$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    if out:
        for line in out.splitlines()[-15:]:
            print(f"  {line}")
    if err and code != 0:
        for line in err.splitlines()[-10:]:
            print(f"  [err] {line}")
    print(f"  exit: {code}\n")
    return out, code


# 1. Install system deps
print("=== Step 1: System dependencies ===")
run("apt-get update -qq && apt-get install -y -qq python3-pip python3-venv postgresql postgresql-client git > /dev/null 2>&1; echo done")

# 2. Start PostgreSQL
print("=== Step 2: PostgreSQL ===")
run("systemctl start postgresql; systemctl enable postgresql; echo pg-started")

# 3. Create DB + user
print("=== Step 3: Create database ===")
run("""su - postgres -c "psql -c \\"SELECT 1 FROM pg_roles WHERE rolname='rfpose'\\" | grep -q 1 || psql -c \\"CREATE USER rfpose WITH PASSWORD 'rfpose';\\""  """)
run("""su - postgres -c "psql -lqt | grep -q rfpose || psql -c \\"CREATE DATABASE rfpose OWNER rfpose;\\""  """)

# 4. Clone / pull repo
print("=== Step 4: Clone repo ===")
run("cd /opt/rfpose && (test -d rf-worldpose/.git && cd rf-worldpose && git pull || git clone https://github.com/tientruongminh/rf-worldpose.git)", timeout=60)

# 5. Run migrations
print("=== Step 5: Run migrations ===")
run("PGPASSWORD=rfpose psql -h localhost -U rfpose -d rfpose -f /opt/rfpose/rf-worldpose/infra/postgres/migrations/001_initial.sql")
run("PGPASSWORD=rfpose psql -h localhost -U rfpose -d rfpose -f /opt/rfpose/rf-worldpose/infra/postgres/migrations/002_training_jobs_portal.sql")

# 6. Setup Python venv + install
print("=== Step 6: Python environment ===")
run("cd /opt/rfpose/rf-worldpose && python3 -m venv .venv && .venv/bin/pip install --upgrade pip -q")
run("cd /opt/rfpose/rf-worldpose && .venv/bin/pip install -e services/api/ -q", timeout=120)

# 7. Create .env
print("=== Step 7: Write .env ===")
env_content = """DATABASE_URL=postgresql://rfpose:rfpose@localhost:5432/rfpose
HPC_LOGIN=eagle.man.poznan.pl
HPC_USER=tiencd1234
HPC_SSH_KEY=/opt/rfpose/.ssh/helios_ed25519
HPC_ACCOUNT=pl0501-01
HPC_PARTITION=
HPC_WORK_DIR=~/pl0501-01/project_data/tien
S3_BUCKET=rfpose
S3_ENDPOINT_URL=http://207.180.243.242:9000
MLFLOW_TRACKING_URI=http://207.180.243.242:5000
"""
run(f"cat > /opt/rfpose/rf-worldpose/.env << 'ENVEOF'\n{env_content}ENVEOF")

# 8. Kill old process if running, then start
print("=== Step 8: Start API ===")
run("pkill -f 'uvicorn rfpose_api' 2>/dev/null; sleep 1; echo killed-old")
run(
    "cd /opt/rfpose/rf-worldpose && "
    "nohup .venv/bin/python -m uvicorn rfpose_api.main:app --host 0.0.0.0 --port 8080 "
    "--env-file .env > /var/log/rfpose-api.log 2>&1 & "
    "sleep 3 && echo started && curl -s http://localhost:8080/health"
)

print("=" * 50)
print("DEPLOY COMPLETE")
print(f"Portal: http://{HOST}:8080/portal")
print(f"API docs: http://{HOST}:8080/docs")
print("=" * 50)

ssh.close()
