from fastapi import APIRouter
from rfpose_api.db.connection import connect
from rfpose_api.schemas.common import RecordingSessionCreate

router = APIRouter(prefix="/api/v1/recording-sessions", tags=["recording-sessions"])

@router.post("")
def create_session(payload: RecordingSessionCreate):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recording_sessions(id, deployment_id, label, metadata, status, started_at)
            VALUES (%s, %s, %s, %s, 'recording', now())
            RETURNING *
            """,
            (payload.id, payload.deployment_id, payload.label, payload.metadata),
        )
        return cur.fetchone()

@router.post("/{session_id}/finish")
def finish_session(session_id: str, bronze_uri: str | None = None):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE recording_sessions SET status='finished', ended_at=now(), bronze_uri=COALESCE(%s, bronze_uri) WHERE id=%s RETURNING *",
            (bronze_uri, session_id),
        )
        return cur.fetchone()
