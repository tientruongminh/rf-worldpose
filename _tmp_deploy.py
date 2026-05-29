import paramiko, sys, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("207.180.243.242", username="root", password="teamKDL123456", timeout=15)

def run(cmd, timeout=120):
    print(f"\n>>> {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out.strip():
        print(out.strip())
    if err.strip():
        print("STDERR:", err.strip())
    return out

# 1. Check .env
run("ls -la /opt/rfpose/rf-worldpose/infra/docker-compose/.env 2>/dev/null || echo NO_ENV")

# 2. Create .env if missing
run("""cat > /opt/rfpose/rf-worldpose/infra/docker-compose/.env << 'ENVEOF'
POSTGRES_USER=rfpose
POSTGRES_PASSWORD=rfpose_dev_2024
POSTGRES_DB=rfpose
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
S3_BUCKET=rfpose
HELIOS_LOGIN=
HELIOS_ACCOUNT=
HELIOS_PARTITION=
ENVEOF
echo ENV_CREATED""")

# 3. Deploy
run("cd /opt/rfpose/rf-worldpose/infra/docker-compose && docker compose up -d --force-recreate 2>&1", timeout=300)

# 4. Wait and check
time.sleep(10)
run("cd /opt/rfpose/rf-worldpose/infra/docker-compose && docker compose ps --format 'table {{.Name}}\t{{.Status}}\t{{.Ports}}' 2>&1")

ssh.close()
print("\n=== Deploy complete ===")
