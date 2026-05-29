"""Data access for dataset_versions table."""
from __future__ import annotations
from rfpose_api.db.connection import connect


def get(dataset_version: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM dataset_versions WHERE id=%s", (dataset_version,))
        return cur.fetchone()


def list_ids(limit: int = 50) -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM dataset_versions ORDER BY created_at DESC LIMIT %s", (limit,))
        return cur.fetchall()


def create(*, id: str, source_sessions, preprocess_version: str,
           teacher_version: str | None, artifact_uri: str, stats: dict,
           quality_report_uri: str | None, created_by: str | None) -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO dataset_versions(id, source_sessions, preprocess_version, teacher_version,
               artifact_uri, stats, quality_report_uri, created_by)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *""",
            (id, source_sessions, preprocess_version, teacher_version,
             artifact_uri, stats, quality_report_uri, created_by),
        )
        return cur.fetchone()
