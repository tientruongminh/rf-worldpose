from fastapi import APIRouter, HTTPException, Query
from rfpose_api.db.connection import connect
from rfpose_api.schemas.common import TrainingJobCreate

router = APIRouter(prefix="/api/v1/training-jobs", tags=["training-jobs"])


@router.get("")
def list_training_jobs(
    status: str | None = Query(None),
    submitted_by: str | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
):
    clauses, params = [], []
    if status:
        clauses.append("status = %s")
        params.append(status)
    if submitted_by:
        clauses.append("submitted_by = %s")
        params.append(submitted_by)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM training_jobs {where} ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params += [limit, offset]

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.execute(f"SELECT count(*) as total FROM training_jobs {where}", params[:-2] or [])
        total = cur.fetchone()["total"]
    return {"items": rows, "total": total, "limit": limit, "offset": offset}


@router.post("")
def create_training_job(payload: TrainingJobCreate):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO training_jobs(id, dataset_version, train_config, backend, submitted_by, status)
            VALUES (%s, %s, %s, %s, %s, 'created')
            RETURNING *
            """,
            (payload.id, payload.dataset_version, payload.train_config,
             payload.backend, payload.submitted_by),
        )
        return cur.fetchone()


@router.get("/{job_id}")
def get_training_job(job_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "training job not found")
        return row


@router.post("/{job_id}/mark-submitted")
def mark_submitted(job_id: str, slurm_job_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE training_jobs
               SET status='submitted', slurm_job_id=%s, submitted_at=now(), updated_at=now()
               WHERE id=%s RETURNING *""",
            (slurm_job_id, job_id),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "training job not found")
        return row
