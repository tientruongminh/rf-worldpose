"""Data access for model_versions table."""
from __future__ import annotations
from rfpose_api.db.connection import connect


def get(model_id: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM model_versions WHERE id=%s", (model_id,))
        return cur.fetchone()


def list_all(limit: int = 100) -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM model_versions ORDER BY created_at DESC LIMIT %s", (limit,))
        return cur.fetchall()


def get_active() -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM model_versions WHERE status='production' LIMIT 1")
        return cur.fetchone()


def create(*, id: str, name: str, dataset_version: str | None, training_job_id: str | None,
           artifact_uri: str, metrics: dict, eval_report_uri: str | None, hash: str | None) -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO model_versions(id, name, dataset_version, training_job_id, artifact_uri, metrics, eval_report_uri, hash)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *""",
            (id, name, dataset_version, training_job_id, artifact_uri, metrics, eval_report_uri, hash),
        )
        return cur.fetchone()


def archive_current_production(model_id: str) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE model_versions SET status='archived' "
            "WHERE name=(SELECT name FROM model_versions WHERE id=%s) AND status='production'",
            (model_id,),
        )


def set_status(model_id: str, status: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE model_versions SET status=%s, promoted_at=now() WHERE id=%s RETURNING *",
            (status, model_id),
        )
        return cur.fetchone()
