"""Data access for training_jobs table."""
from __future__ import annotations
from rfpose_api.db.connection import connect


def get(job_id: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        return cur.fetchone()


def list_all(*, status: str | None = None, submitted_by: str | None = None,
             limit: int = 100, offset: int = 0) -> tuple[list[dict], int]:
    clauses, params = [], []
    if status:
        clauses.append("status = %s")
        params.append(status)
    if submitted_by:
        clauses.append("submitted_by = %s")
        params.append(submitted_by)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT * FROM training_jobs {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
            params + [limit, offset],
        )
        rows = cur.fetchall()
        cur.execute(f"SELECT count(*) as total FROM training_jobs {where}", params or [])
        total = cur.fetchone()["total"]
    return rows, total


def stats() -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT count(*) as total,
                   count(*) FILTER (WHERE status='running') as running,
                   count(*) FILTER (WHERE status='submitted') as submitted,
                   count(*) FILTER (WHERE status='completed') as completed,
                   count(*) FILTER (WHERE status='failed') as failed
            FROM training_jobs
        """)
        return cur.fetchone()


def create(*, job_id: str, dataset_version: str, train_config: str,
           backend: str = "eagle-slurm", submitted_by: str = "portal") -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO training_jobs(id, dataset_version, train_config, backend, submitted_by, status)
               VALUES (%s, %s, %s, %s, %s, 'created') RETURNING *""",
            (job_id, dataset_version, train_config, backend, submitted_by),
        )
        return cur.fetchone()


def create_submitted(*, job_id: str, dataset_version: str, train_config: str,
                     backend: str, submitted_by: str, slurm_job_id: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO training_jobs(id, dataset_version, train_config, backend, submitted_by, status, slurm_job_id, submitted_at)
               VALUES (%s, %s, %s, %s, %s, 'submitted', %s, now())
               ON CONFLICT (id) DO NOTHING RETURNING *""",
            (job_id, dataset_version, train_config, backend, submitted_by, slurm_job_id),
        )
        return cur.fetchone()


def mark_submitted(job_id: str, slurm_job_id: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE training_jobs
               SET status='submitted', slurm_job_id=%s, submitted_at=now(), updated_at=now()
               WHERE id=%s RETURNING *""",
            (slurm_job_id, job_id),
        )
        return cur.fetchone()


def update_status(job_id: str, *, slurm_state: str, status: str,
                  set_started: bool = False, set_finished: bool = False) -> dict | None:
    set_parts = ["slurm_state=%s", "status=%s", "updated_at=now()"]
    vals: list = [slurm_state, status]
    if set_started:
        set_parts.append("started_at=now()")
    if set_finished:
        set_parts.append("finished_at=COALESCE(finished_at, now())")
    vals.append(job_id)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"UPDATE training_jobs SET {', '.join(set_parts)} WHERE id=%s RETURNING *", vals)
        return cur.fetchone()


def mark_cancelled(job_id: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE training_jobs
               SET status='cancelled', slurm_state='CANCELLED', finished_at=now(), updated_at=now()
               WHERE id=%s RETURNING *""",
            (job_id,),
        )
        return cur.fetchone()


def list_active() -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, slurm_job_id, status FROM training_jobs WHERE status = ANY(%s) AND slurm_job_id IS NOT NULL",
            (["submitted", "running"],),
        )
        return cur.fetchall()
