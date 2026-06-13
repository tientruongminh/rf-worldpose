from fastapi import APIRouter, HTTPException, Query

from rfpose_api.repositories import training_jobs as job_repo
from rfpose_api.schemas.common import TrainingJobCreate

router = APIRouter(prefix="/api/v1/training-jobs", tags=["training-jobs"])


@router.get("/stats")
def job_stats():
    return job_repo.stats()


@router.get("")
def list_training_jobs(
    status: str | None = Query(None),
    submitted_by: str | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
):
    rows, total = job_repo.list_all(
        status=status,
        submitted_by=submitted_by,
        limit=limit,
        offset=offset,
    )
    return {"items": rows, "total": total, "limit": limit, "offset": offset}


@router.post("")
def create_training_job(payload: TrainingJobCreate):
    return job_repo.create(
        job_id=payload.id,
        dataset_version=payload.dataset_version,
        train_config=payload.train_config,
        backend=payload.backend,
        submitted_by=payload.submitted_by or "api",
    )


@router.get("/{job_id}")
def get_training_job(job_id: str):
    row = job_repo.get(job_id)
    if not row:
        raise HTTPException(404, "training job not found")
    return row


@router.post("/{job_id}/mark-submitted")
def mark_submitted(job_id: str, slurm_job_id: str):
    row = job_repo.mark_submitted(job_id, slurm_job_id)
    if not row:
        raise HTTPException(404, "training job not found")
    return row
