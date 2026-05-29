"""REST API for Datasets — thin router."""
from fastapi import APIRouter, HTTPException
from rfpose_api.schemas.common import DatasetVersionCreate
from rfpose_api.repositories import datasets as dataset_repo

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.get("")
def list_datasets():
    return dataset_repo.list_ids()


@router.post("")
def create_dataset(payload: DatasetVersionCreate):
    return dataset_repo.create(
        id=payload.id, source_sessions=payload.source_sessions,
        preprocess_version=payload.preprocess_version,
        teacher_version=payload.teacher_version,
        artifact_uri=payload.artifact_uri, stats=payload.stats,
        quality_report_uri=payload.quality_report_uri,
        created_by=payload.created_by,
    )


@router.get("/{dataset_version}")
def get_dataset(dataset_version: str):
    row = dataset_repo.get(dataset_version)
    if not row:
        raise HTTPException(404, "dataset not found")
    return row
