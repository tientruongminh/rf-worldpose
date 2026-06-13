"""HPC (Slurm) orchestration service — single source of truth for job lifecycle.

Eliminates duplication between routers — single entry point for HPC operations.
"""
from __future__ import annotations

import logging
from typing import Any

from rfpose_api.config import settings
from rfpose_api.repositories import training_jobs as job_repo
from rfpose_eagle.registry import get_preset
from rfpose_eagle.submit import EagleJobSpec, submit_training_job
from rfpose_api.services.ssh_executor import (
    submit_script,
    test_connection, list_remote_scripts,
    slurm_status, slurm_job_detail, fetch_slurm_logs,
    cancel_job,
)

log = logging.getLogger(__name__)

SLURM_STATUS_MAP = {
    "PENDING": "submitted",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
}

SLURM_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}

_login = settings.hpc_ssh_target
_ssh_key = settings.hpc_ssh_key
_work_dir = settings.hpc_work_dir


def map_slurm_state(slurm_state: str, current_status: str) -> str:
    return SLURM_STATUS_MAP.get(slurm_state, current_status)


def connection_check() -> dict:
    return test_connection(_login, ssh_key=_ssh_key)


def list_scripts() -> list[str]:
    return list_remote_scripts(_login, ssh_key=_ssh_key, remote_dir=_work_dir)


def submit_existing_script(script_name: str) -> str:
    """Submit a pre-existing .sbatch on the HPC node. Returns Slurm job ID."""
    return submit_script(login=_login, ssh_key=_ssh_key, remote_dir=_work_dir, script_name=script_name)


def submit_job(job: dict, *, preset_config: dict | None = None, dry_run: bool = False) -> str:
    """Render sbatch from job + optional preset, submit to HPC. Returns sbatch text (dry_run) or Slurm ID."""
    preset = get_preset(job["train_config"])
    spec = EagleJobSpec(
        job_id=job["id"],
        dataset_version=job.get("dataset_version") or preset.dataset_version,
        train_config=job["train_config"],
        partition=settings.hpc_partition,
        gpus=preset.gpus,
        cpus=preset.cpus,
        mem=preset.mem,
        time_limit=preset.time_limit,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
    )
    return submit_training_job(
        spec,
        ssh_host=_login,
        ssh_key=_ssh_key,
        remote_dir=_work_dir,
        repo_root="/app",
        sync=not dry_run,
        dry_run=dry_run,
    )


def submit_and_record(job: dict, *, preset_config: dict | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Submit job and update DB. Returns {"dry_run": ..., "sbatch": ...} or updated job row."""
    result = submit_job(job, preset_config=preset_config, dry_run=dry_run)
    if dry_run:
        return {"dry_run": True, "sbatch": result}
    updated = job_repo.mark_submitted(job["id"], result)
    return {"dry_run": False, "job": updated, "slurm_job_id": result}


def refresh_job_status(job_id: str) -> dict | None:
    """Query Slurm for latest status, update DB. Single source of truth."""
    job = job_repo.get(job_id)
    if not job or not job.get("slurm_job_id"):
        return job

    raw = slurm_status(_login, job["slurm_job_id"], ssh_key=_ssh_key)
    parts = raw.split("|") if raw else []
    slurm_state = parts[1] if len(parts) > 1 else "UNKNOWN"
    new_status = map_slurm_state(slurm_state, job["status"])

    return job_repo.update_status(
        job_id,
        slurm_state=slurm_state,
        status=new_status,
        set_started=(new_status == "running" and not job.get("started_at")),
        set_finished=(slurm_state in SLURM_TERMINAL and not job.get("finished_at")),
    )


def cancel(job_id: str) -> dict | None:
    """Cancel Slurm job and update DB."""
    job = job_repo.get(job_id)
    if not job or not job.get("slurm_job_id"):
        return job
    cancel_job(_login, job["slurm_job_id"], ssh_key=_ssh_key)
    return job_repo.mark_cancelled(job_id)


def get_logs(job: dict) -> dict:
    """Fetch stdout/stderr from HPC."""
    return fetch_slurm_logs(_login, job["slurm_job_id"], ssh_key=_ssh_key, remote_dir=_work_dir)


def get_detail(job: dict) -> dict:
    """Fetch extended sacct info."""
    return slurm_job_detail(_login, job["slurm_job_id"], ssh_key=_ssh_key)


def poll_active_jobs() -> int:
    """Poll all active jobs. Returns count of jobs polled. Used by status_poller."""
    active = job_repo.list_active()
    if not active:
        return 0

    for job in active:
        try:
            refresh_job_status(job["id"])
            log.info("polled job %s", job["id"])
        except Exception:
            log.exception("failed to poll job %s", job["id"])

    return len(active)
