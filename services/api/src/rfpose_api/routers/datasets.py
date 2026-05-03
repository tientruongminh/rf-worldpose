from fastapi import APIRouter
from rfpose_api.db.connection import connect
from rfpose_api.schemas.common import DatasetVersionCreate

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

@router.post("")
def create_dataset(payload: DatasetVersionCreate):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dataset_versions(id, source_sessions, preprocess_version, teacher_version, artifact_uri, stats, quality_report_uri, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (payload.id, payload.source_sessions, payload.preprocess_version, payload.teacher_version, payload.artifact_uri, payload.stats, payload.quality_report_uri, payload.created_by),
        )
        return cur.fetchone()

@router.get("/{dataset_version}")
def get_dataset(dataset_version: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM dataset_versions WHERE id=%s", (dataset_version,))
        return cur.fetchone()
