"""REST API for Eagle HPC operations — submit, status, cancel training jobs."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException

from rfpose_api.config import settings
from rfpose_api.db.connection import connect
from rfpose_eagle.submit import (
    EagleJobSpec, render_sbatch, submit_training_job, sync_code,
    resolve_train_module, _ssh_opts,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hpc", tags=["hpc"])

CONFIGS_DIR = Path("/app/ml/configs")
REPO_ROOT = "/app"


def _ssh_target() -> str:
    return settings.hpc_ssh_target


def _ssh_key() -> str:
    return settings.hpc_ssh_key


def _run_ssh(cmd: str, timeout: int = 15) -> str:
    opts = _ssh_opts(_ssh_key())
    return subprocess.check_output(
        ["ssh", *opts, _ssh_target(), cmd], text=True, timeout=timeout,
    ).strip()


@router.get("/connection-test")
def connection_test():
    try:
        out = _run_ssh("echo ok && hostname")
        return {"ok": True, "output": out}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/configs")
def list_configs():
    """Scan ml/configs/*.yaml and return available training configs."""
    configs = []
    if CONFIGS_DIR.exists():
        for f in sorted(CONFIGS_DIR.glob("*.yaml")):
            name = f.stem
            module = resolve_train_module(name)
            configs.append({
                "name": name,
                "module": module,
                "file": f.name,
            })
    return configs


@router.post("/training-jobs/{job_id}/submit")
def submit_job(job_id: str, dry_run: bool = False):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        job = cur.fetchone()
        if not job:
            raise HTTPException(404, "training job not found")

    config_name = job["train_config"]
    spec = EagleJobSpec(
        job_id=job_id,
        config_name=config_name,
        dataset_version=job.get("dataset_version", "rfpose-multitask-v1"),
        partition=settings.hpc_partition or "tesla",
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
    )

    try:
        result = submit_training_job(
            spec,
            ssh_host="eagle",
            ssh_key=_ssh_key(),
            remote_dir=settings.hpc_work_dir,
            repo_root=REPO_ROOT,
            sync=True,
            dry_run=dry_run,
        )
    except Exception as exc:
        log.exception("submit failed for job %s", job_id)
        raise HTTPException(502, f"submit failed: {exc}")

    if dry_run:
        return {"dry_run": True, "sbatch": result}

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE training_jobs
               SET status='submitted', slurm_job_id=%s, submitted_at=now(), updated_at=now()
               WHERE id=%s RETURNING *""",
            (result, job_id),
        )
        return cur.fetchone()


@router.post("/training-jobs/{job_id}/refresh-status")
def refresh_status(job_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        job = cur.fetchone()
        if not job:
            raise HTTPException(404, "training job not found")
        if not job.get("slurm_job_id"):
            raise HTTPException(400, "job has no slurm_job_id")

    try:
        raw = _run_ssh(
            f"sacct -j {job['slurm_job_id']} --format=JobID,State,Elapsed,ExitCode "
            f"--parsable2 --noheader | head -1"
        )
    except Exception as exc:
        raise HTTPException(502, f"failed to reach Slurm: {exc}")

    parts = raw.split("|") if raw else []
    slurm_state = parts[1] if len(parts) > 1 else "UNKNOWN"

    status_map = {
        "PENDING": "submitted", "RUNNING": "running", "COMPLETING": "running",
        "COMPLETED": "completed", "FAILED": "failed", "CANCELLED": "cancelled",
        "TIMEOUT": "failed", "OUT_OF_MEMORY": "failed", "NODE_FAIL": "failed",
    }
    new_status = status_map.get(slurm_state, job["status"])

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE training_jobs
               SET slurm_state=%s, status=%s, updated_at=now()
               WHERE id=%s RETURNING *""",
            (slurm_state, new_status, job_id),
        )
        return cur.fetchone()


@router.post("/training-jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        job = cur.fetchone()
        if not job:
            raise HTTPException(404, "training job not found")
        if not job.get("slurm_job_id"):
            raise HTTPException(400, "job has no slurm_job_id")

    try:
        _run_ssh(f"scancel {job['slurm_job_id']}")
    except Exception as exc:
        raise HTTPException(502, f"cancel failed: {exc}")

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE training_jobs
               SET status='cancelled', slurm_state='CANCELLED',
                   finished_at=now(), updated_at=now()
               WHERE id=%s RETURNING *""",
            (job_id,),
        )
        return cur.fetchone()
