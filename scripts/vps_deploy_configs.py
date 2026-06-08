"""Deploy Config Registry: run migration 003 on VPS and restart API."""
import os
import sys

import paramiko

VPS = os.environ.get("VPS_HOST", "207.180.243.242")
USER = os.environ.get("VPS_USER", "root")
PASSWORD = os.environ.get("VPS_PASSWORD", "")

def run(ssh, cmd, desc=""):
    print(f"\n>>> {desc or cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=60)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out.strip():
        print(out.strip())
    if err.strip():
        print(f"STDERR: {err.strip()}")
    return out.strip()

def main():
    if not PASSWORD:
        print("ERROR: VPS_PASSWORD env var is not set")
        sys.exit(1)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS, username=USER, password=PASSWORD, timeout=15)

    pg_container = run(ssh, "docker ps --format '{{.Names}}' | grep -i postgres", "Find postgres container")
    if not pg_container:
        print("ERROR: No postgres container found!")
        ssh.close()
        return
    pg_container = pg_container.strip().split('\n')[0]
    print(f"  Using container: {pg_container}")

    run(ssh, "cd /opt/rfpose/rf-worldpose && git pull origin main", "Pull latest code")

    migration = f"""docker exec {pg_container} psql -U rfpose -d rfpose -c "
CREATE TABLE IF NOT EXISTS training_configs (
    id            TEXT PRIMARY KEY,
    label         TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    script_path   TEXT NOT NULL DEFAULT 'training/train.py',
    git_repo      TEXT NOT NULL DEFAULT 'https://github.com/tientruongminh/rf-worldpose',
    git_branch    TEXT NOT NULL DEFAULT 'main',
    dataset_hint  TEXT NOT NULL DEFAULT '',
    hyperparams   TEXT NOT NULL DEFAULT '',
    requirements  TEXT NOT NULL DEFAULT 'torch mlflow numpy h5py',
    created_by    TEXT NOT NULL DEFAULT 'system',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
"
"""
    run(ssh, migration.strip(), "Create training_configs table")

    seed_sql = """INSERT INTO training_configs (id, label, description, script_path, dataset_hint, hyperparams, created_by) VALUES
('wipose-action-baseline','Wi-Pose Action Recognition (baseline)','Baseline action classification on 12 activities','training/train_wipose_action.py','wipose-v1','epochs=50
batch_size=128
lr=3e-4
num_nodes=9
window_frames=30
n_subcarriers=30
channels=5
num_classes=12
num_keypoints=18','system'),
('wipose-pose2d','Wi-Pose 2D Pose Estimation','Joint regression from CSI using SkeletonPoints','training/train_wipose_pose.py','wipose-v1','epochs=80
batch_size=64
lr=1e-4
num_nodes=9
window_frames=30
n_subcarriers=30
channels=5
num_keypoints=18
pose_weight=1.0','system'),
('wiar-action-baseline','WiAR 16-Activity Recognition (baseline)','Baseline activity recognition on 16 activities','training/train_wiar_action.py','wiar-v1','epochs=50
batch_size=128
lr=3e-4
num_nodes=9
window_frames=30
n_subcarriers=30
channels=1
num_classes=16','system'),
('wiar-action-lora','WiAR LoRA Fine-tune','Fine-tune pretrained model with LoRA adapters','training/train_wiar_lora.py','wiar-v1','epochs=30
batch_size=64
lr=1e-4
num_nodes=9
window_frames=30
n_subcarriers=30
channels=1
num_classes=16
lora_rank=8
lora_alpha=16','system')
ON CONFLICT (id) DO NOTHING;"""

    sftp = ssh.open_sftp()
    sftp.open("/tmp/seed_configs.sql", "w").write(seed_sql)
    sftp.close()

    run(ssh, f"docker cp /tmp/seed_configs.sql {pg_container}:/tmp/seed_configs.sql", "Copy seed SQL")
    run(ssh, f"docker exec {pg_container} psql -U rfpose -d rfpose -f /tmp/seed_configs.sql", "Seed default presets")

    run(ssh, f"docker exec {pg_container} psql -U rfpose -d rfpose -c 'SELECT id, label, dataset_hint FROM training_configs;'",
        "Verify configs")

    run(ssh, "cd /opt/rfpose/rf-worldpose/infra/docker-compose && docker compose restart api",
        "Restart API container")

    run(ssh, "sleep 5 && curl -s http://localhost:8080/portal/configs | grep -o 'Config Registry' | head -1",
        "Test Config Registry page")

    ssh.close()
    print("\n=== Deploy complete! ===")
    print("Config Registry: http://207.180.243.242:8080/portal/configs")
    print("Submit Job:      http://207.180.243.242:8080/portal/submit")

if __name__ == "__main__":
    main()
