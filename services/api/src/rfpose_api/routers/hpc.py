"""REST API for Eagle HPC operations: presets, submit, status, cancel, logs."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from rfpose_api.repositories import training_jobs as job_repo
from rfpose_api.services import hpc as hpc_service
from rfpose_eagle.registry import get_preset, list_presets, resolve_train_module

router = APIRouter(prefix="/api/v1/hpc", tags=["hpc"])

CONFIGS_DIR = Path("/app/ml/configs")


@router.get("/connection-test")
def connection_test():
    return hpc_service.connection_check()


@router.get("/configs")
def list_configs():
    """Return registered model presets plus any YAML configs found in ml/configs."""
    registered = {item["config_name"]: item for item in list_presets()}
    config_names = set(registered)
    if CONFIGS_DIR.exists():
        config_names.update(path.stem for path in CONFIGS_DIR.glob("*.yaml"))

    items = []
    for name in sorted(config_names):
        preset = registered.get(name) or get_preset(name).to_dict()
        items.append({
            "name": name,
            "module": resolve_train_module(name),
            "file": f"{name}.yaml",
            "registered": name in registered,
            "recommended": preset.get("recommended", False),
            "model_family": preset.get("model_family", "custom"),
            "task": preset.get("task", "custom"),
            "dataset_version": preset.get("dataset_version", "rfpose-unified-v2"),
            "description": preset.get("description", ""),
        })
    return items


@router.post("/training-jobs/{job_id}/submit")
def submit_job(job_id: str, dry_run: bool = False):
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(404, "training job not found")
    try:
        return hpc_service.submit_and_record(job, dry_run=dry_run)
    except Exception as exc:
        raise HTTPException(502, f"submit failed: {exc}") from exc


@router.post("/training-jobs/{job_id}/refresh-status")
def refresh_status(job_id: str):
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(404, "training job not found")
    if not job.get("slurm_job_id"):
        raise HTTPException(400, "job has no slurm_job_id")
    try:
        return hpc_service.refresh_job_status(job_id)
    except Exception as exc:
        raise HTTPException(502, f"failed to reach Slurm: {exc}") from exc


@router.get("/training-jobs/{job_id}/detail")
def job_detail(job_id: str):
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(404, "training job not found")
    if not job.get("slurm_job_id"):
        raise HTTPException(400, "job has no slurm_job_id")
    return hpc_service.get_detail(job)


@router.get("/training-jobs/{job_id}/logs")
def job_logs(job_id: str):
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(404, "training job not found")
    if not job.get("slurm_job_id"):
        raise HTTPException(400, "job has no slurm_job_id")
    return hpc_service.get_logs(job)


@router.post("/training-jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(404, "training job not found")
    if not job.get("slurm_job_id"):
        raise HTTPException(400, "job has no slurm_job_id")
    try:
        return hpc_service.cancel(job_id)
    except Exception as exc:
        raise HTTPException(502, f"cancel failed: {exc}") from exc
