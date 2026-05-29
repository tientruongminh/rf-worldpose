"""Data access for recording_sessions table."""
from __future__ import annotations

from rfpose_api.db.connection import connect


def create(
    *, id: str, deployment_id: str, label: str, metadata: str | None = None
) -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO recording_sessions(id, deployment_id, label, metadata, status, started_at)
               VALUES (%s, %s, %s, %s, 'recording', now()) RETURNING *""",
            (id, deployment_id, label, metadata),
        )
        return cur.fetchone()


def finish(session_id: str, bronze_uri: str | None = None) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE recording_sessions
               SET status='finished', ended_at=now(), bronze_uri=COALESCE(%s, bronze_uri)
               WHERE id=%s RETURNING *""",
            (bronze_uri, session_id),
        )
        return cur.fetchone()
