"""Deploy Config Registry: run migration 003 on VPS and restart API."""
import paramiko

VPS = "207.180.243.242"
USER = "root"
PASSWORD = "teamKDL123456"

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
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS, username=USER, password=PASSWORD, timeout=15)

    run(ssh, "cd /opt/rfpose/rf-worldpose && git pull origin main", "Pull latest code")

    migration = """
    psql -U rfpose -d rfpose -c "
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
    run(ssh, f'docker exec rfpose-postgres {migration.strip()}', "Create training_configs table")

    seed = r"""psql -U rfpose -d rfpose -c "
    INSERT INTO training_configs (id, label, description, script_path, dataset_hint, hyperparams, created_by)
    VALUES
      ('wipose-action-baseline',
       'Wi-Pose Action Recognition (baseline)',
       'Baseline action classification on 12 activities using 9-node CSI',
       'training/train_wipose_action.py',
       'wipose-v1',
       E'epochs=50\nbatch_size=128\nlr=3e-4\nnum_nodes=9\nwindow_frames=30\nn_subcarriers=30\nchannels=5\nnum_classes=12\nnum_keypoints=18',
       'system'),
      ('wipose-pose2d',
       'Wi-Pose 2D Pose Estimation',
       'Joint regression from CSI amplitude using SkeletonPoints labels',
       'training/train_wipose_pose.py',
       'wipose-v1',
       E'epochs=80\nbatch_size=64\nlr=1e-4\nnum_nodes=9\nwindow_frames=30\nn_subcarriers=30\nchannels=5\nnum_keypoints=18\npose_weight=1.0',
       'system'),
      ('wiar-action-baseline',
       'WiAR 16-Activity Recognition (baseline)',
       'Baseline activity recognition on 16 activities using Intel 5300 CSI',
       'training/train_wiar_action.py',
       'wiar-v1',
       E'epochs=50\nbatch_size=128\nlr=3e-4\nnum_nodes=9\nwindow_frames=30\nn_subcarriers=30\nchannels=1\nnum_classes=16',
       'system'),
      ('wiar-action-lora',
       'WiAR LoRA Fine-tune',
       'Fine-tune pretrained model on WiAR with LoRA adapters',
       'training/train_wiar_lora.py',
       'wiar-v1',
       E'epochs=30\nbatch_size=64\nlr=1e-4\nnum_nodes=9\nwindow_frames=30\nn_subcarriers=30\nchannels=1\nnum_classes=16\nlora_rank=8\nlora_alpha=16',
       'system')
    ON CONFLICT (id) DO NOTHING;
    "
    """
    run(ssh, f'docker exec rfpose-postgres {seed.strip()}', "Seed default presets")

    run(ssh, "docker exec rfpose-postgres psql -U rfpose -d rfpose -c 'SELECT id, label, dataset_hint FROM training_configs;'",
        "Verify configs")

    run(ssh, "cd /opt/rfpose/rf-worldpose/infra/docker-compose && docker compose restart api",
        "Restart API container")

    run(ssh, "sleep 3 && curl -s http://localhost:8080/portal/configs | head -20",
        "Test Config Registry page")

    ssh.close()
    print("\n=== Deploy complete! ===")
    print("Config Registry: http://207.180.243.242:8080/portal/configs")
    print("Submit Job:      http://207.180.243.242:8080/portal/submit")

if __name__ == "__main__":
    main()
