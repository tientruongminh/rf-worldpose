"""Dagster job: auto-submit HPC training when Gold dataset is ready.

Triggered after the `gold_dataset` asset materializes successfully.
Reads training configs from Postgres, renders sbatch, and submits to HPC via SSH.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from dagster import job, op, In, Out, Output, OpExecutionContext

DB_URL = os.environ.get("DATABASE_URL", "postgresql://rfpose:rfpose@postgres:5432/rfpose")


def _get_db():
    import psycopg
    return psycopg.connect(DB_URL, row_factory=psycopg.rows.dict_row)


@op(out={"configs": Out(list)})
def load_training_configs(context: OpExecutionContext):
    """Fetch active training configs from the registry that match the current dataset."""
    dataset_version = os.environ.get("RFPOSE_DATASET_VERSION", "")
    with _get_db() as conn, conn.cursor() as cur:
        if dataset_version:
            cur.execute(
                "SELECT * FROM training_configs WHERE dataset_hint = %s OR dataset_hint IS NULL ORDER BY id",
                (dataset_version,),
            )
        else:
            cur.execute("SELECT * FROM training_configs ORDER BY id")
        configs = cur.fetchall()

    context.log.info("Found %d training configs", len(configs))
    return Output(configs, metadata={"count": len(configs)})


@op(ins={"configs": In(list)})
def submit_hpc_jobs(context: OpExecutionContext, configs: list):
    """Render sbatch for each config and submit to HPC via SSH."""
    hpc_login = os.environ.get("HPC_LOGIN", "")
    hpc_user = os.environ.get("HPC_USER", "")
    hpc_ssh_key = os.environ.get("HPC_SSH_KEY", "")
    hpc_account = os.environ.get("HPC_ACCOUNT", "")
    hpc_partition = os.environ.get("HPC_PARTITION", "")
    hpc_work_dir = os.environ.get("HPC_WORK_DIR", "~/rfpose-jobs")
    s3_bucket = os.environ.get("S3_BUCKET", "rfpose")
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL", "http://minio:9000")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    dataset_version = os.environ.get("RFPOSE_DATASET_VERSION", "rfpose-local-stub")

    if not hpc_login:
        context.log.warning("HPC_LOGIN not set — skipping job submission")
        return

    ssh_target = f"{hpc_user}@{hpc_login}" if hpc_user else hpc_login

    template_path = Path(__file__).resolve().parents[4] / "helios_runner" / "templates" / "train_gh200.sbatch"
    if not template_path.exists():
        context.log.error("Sbatch template not found at %s", template_path)
        return

    template = template_path.read_text()
    submitted = []

    for cfg in configs:
        job_id = f"dagster-{cfg['id']}-{dataset_version}"
        hyperparams = cfg.get("hyperparams") or {}
        if isinstance(hyperparams, str):
            hyperparams = json.loads(hyperparams)

        train_config_env = "\n".join(f"export {k}={v}" for k, v in hyperparams.items())

        rendered = template
        replacements = {
            "{{ partition }}": hpc_partition,
            "{{ account }}": hpc_account,
            '{{ time_limit | default("24:00:00") }}': "24:00:00",
            "{{ dataset_version }}": dataset_version,
            "{{ train_config }}": train_config_env,
            "{{ s3_bucket }}": s3_bucket,
            "{{ s3_endpoint_url }}": s3_endpoint,
            "{{ mlflow_tracking_uri }}": mlflow_uri,
            "{{ script_path }}": cfg.get("script_path", "ml/rfpose/training/train.py"),
            "{{ git_repo }}": cfg.get("git_repo", "https://github.com/tientruongminh/rf-worldpose"),
            "{{ git_branch }}": cfg.get("git_branch", "main"),
        }
        for old, new in replacements.items():
            rendered = rendered.replace(old, str(new))

        script_name = f"{job_id}.sbatch"
        local_tmp = Path(os.environ.get("TMPDIR", "/tmp")) / script_name
        local_tmp.write_text(rendered)

        ssh_opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
        if hpc_ssh_key:
            ssh_opts += ["-i", hpc_ssh_key]

        try:
            subprocess.run(
                ["ssh", *ssh_opts, ssh_target, "mkdir", "-p", hpc_work_dir, f"{hpc_work_dir}/logs"],
                check=True, timeout=30,
            )
            subprocess.run(
                ["scp", *ssh_opts, str(local_tmp), f"{ssh_target}:{hpc_work_dir}/{script_name}"],
                check=True, timeout=60,
            )
            slurm_id = subprocess.check_output(
                ["ssh", *ssh_opts, ssh_target, f"cd {hpc_work_dir} && sbatch --parsable {script_name}"],
                text=True, timeout=30,
            ).strip()

            context.log.info("Submitted %s → Slurm ID %s", cfg["label"], slurm_id)
            submitted.append({"config": cfg["label"], "slurm_id": slurm_id})

            with _get_db() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO training_jobs (id, dataset_version, train_config, backend, submitted_by, status, slurm_job_id, submitted_at)
                       VALUES (%s, %s, %s, 'eagle-slurm', 'dagster', 'submitted', %s, now())
                       ON CONFLICT (id) DO UPDATE SET slurm_job_id = EXCLUDED.slurm_job_id, status = 'submitted', submitted_at = now()""",
                    (job_id, dataset_version, json.dumps(cfg), slurm_id),
                )
                conn.commit()

        except Exception as exc:
            context.log.error("Failed to submit %s: %s", cfg["label"], exc)

    context.log.info("Submitted %d/%d jobs", len(submitted), len(configs))


@job(description="Submit all matching training configs to HPC after Gold dataset is ready")
def auto_train_on_gold():
    configs = load_training_configs()
    submit_hpc_jobs(configs)
