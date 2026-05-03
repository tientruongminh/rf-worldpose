from fastapi import APIRouter, HTTPException
from rfpose_api.db.connection import connect
from rfpose_api.schemas.common import ModelVersionCreate

router = APIRouter(prefix="/api/v1/models", tags=["models"])

@router.post("")
def create_model(payload: ModelVersionCreate):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_versions(id, name, dataset_version, training_job_id, artifact_uri, metrics, eval_report_uri, hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (payload.id, payload.name, payload.dataset_version, payload.training_job_id, payload.artifact_uri, payload.metrics, payload.eval_report_uri, payload.hash),
        )
        return cur.fetchone()

@router.post("/{model_id}/promote")
def promote_model(model_id: str, status: str = "production"):
    if status not in {"staging", "production", "archived", "rollback"}:
        raise HTTPException(400, "invalid status")
    with connect() as conn, conn.cursor() as cur:
        if status == "production":
            cur.execute("UPDATE model_versions SET status='archived' WHERE name=(SELECT name FROM model_versions WHERE id=%s) AND status='production'", (model_id,))
        cur.execute("UPDATE model_versions SET status=%s, promoted_at=now() WHERE id=%s RETURNING *", (status, model_id))
        return cur.fetchone()

@router.post("/{model_id}/rollback")
def rollback_model(model_id: str):
    return promote_model(model_id, status="rollback")
