"""REST API for HPC operations — thin router, delegates to hpc service."""
from fastapi import APIRouter, HTTPException
from rfpose_api.services import hpc
from rfpose_api.repositories import training_jobs as job_repo

router = APIRouter(prefix="/api/v1/hpc", tags=["hpc"])


@router.get("/connection-test")
def connection_test():
    return hpc.connection_check()


@router.get("/remote-scripts")
def remote_scripts():
    return {"work_dir": hpc._work_dir, "scripts": hpc.list_scripts()}


@router.post("/submit-script")
def submit_remote_script(script_name: str):
    try:
        slurm_id = hpc.submit_existing_script(script_name)
    except Exception as exc:
        raise HTTPException(502, f"submit failed: {exc}")
    return {"slurm_job_id": slurm_id, "script": script_name}


@router.post("/training-jobs/{job_id}/submit")
def submit_job(job_id: str, dry_run: bool = True):
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(404, "training job not found")
    try:
        return hpc.submit_and_record(job, dry_run=dry_run)
    except Exception as exc:
        raise HTTPException(502, f"submit failed: {exc}")


@router.post("/training-jobs/{job_id}/refresh-status")
def refresh_job_status(job_id: str):
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(404, "training job not found")
    if not job.get("slurm_job_id"):
        raise HTTPException(400, "job has no slurm_job_id — not yet submitted")
    try:
        return hpc.refresh_job_status(job_id)
    except Exception as exc:
        raise HTTPException(502, f"failed to reach Slurm: {exc}")


@router.post("/training-jobs/{job_id}/cancel")
def cancel_slurm_job(job_id: str):
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(404, "training job not found")
    if not job.get("slurm_job_id"):
        raise HTTPException(400, "job has no slurm_job_id")
    try:
        return hpc.cancel(job_id)
    except Exception as exc:
        raise HTTPException(502, f"cancel failed: {exc}")
