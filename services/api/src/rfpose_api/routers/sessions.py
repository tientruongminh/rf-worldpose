"""REST API for Recording Sessions — thin router."""
from fastapi import APIRouter, HTTPException
from rfpose_api.schemas.common import RecordingSessionCreate
from rfpose_api.repositories import sessions as session_repo

router = APIRouter(prefix="/api/v1/recording-sessions", tags=["recording-sessions"])


@router.post("")
def create_session(payload: RecordingSessionCreate):
    return session_repo.create(
        id=payload.id, deployment_id=payload.deployment_id,
        label=payload.label, metadata=payload.metadata,
    )


@router.post("/{session_id}/finish")
def finish_session(session_id: str, bronze_uri: str | None = None):
    row = session_repo.finish(session_id, bronze_uri)
    if not row:
        raise HTTPException(404, "session not found")
    return row
