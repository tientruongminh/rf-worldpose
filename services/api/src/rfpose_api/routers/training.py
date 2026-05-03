from fastapi import APIRouter
from rfpose_api.db.connection import connect
from rfpose_api.schemas.common import TrainingJobCreate

router = APIRouter(prefix="/api/v1/training-jobs", tags=["training-jobs"])

@router.post("")
def create_training_job(payload: TrainingJobCreate):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO training_jobs(id, dataset_version, train_config, backend, status)
            VALUES (%s, %s, %s, %s, 'created')
            RETURNING *
            """,
            (payload.id, payload.dataset_version, payload.train_config, payload.backend),
        )
        return cur.fetchone()

@router.get("/{job_id}")
def get_training_job(job_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        return cur.fetchone()

@router.post("/{job_id}/mark-submitted")
def mark_submitted(job_id: str, slurm_job_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE training_jobs SET status='submitted', slurm_job_id=%s, submitted_at=now() WHERE id=%s RETURNING *",
            (slurm_job_id, job_id),
        )
        return cur.fetchone()
