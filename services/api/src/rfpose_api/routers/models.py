"""REST API for Model Registry — thin router, delegates to model_service."""
from fastapi import APIRouter, HTTPException
from rfpose_api.schemas.common import ModelVersionCreate
from rfpose_api.services import models as model_service

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("")
def api_list_models():
    return model_service.list_models()


@router.get("/{model_id}")
def api_get_model(model_id: str):
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    return model


@router.post("")
def api_create_model(payload: ModelVersionCreate):
    from rfpose_api.repositories import models as repo
    return repo.create(
        id=payload.id, name=payload.name, dataset_version=payload.dataset_version,
        training_job_id=payload.training_job_id, artifact_uri=payload.artifact_uri,
        metrics=payload.metrics, eval_report_uri=payload.eval_report_uri, hash=payload.hash,
    )


@router.post("/{model_id}/promote")
def api_promote(model_id: str, status: str = "production"):
    try:
        result = model_service.promote(model_id, status)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not result:
        raise HTTPException(404, "Model not found")
    return result


@router.post("/{model_id}/deploy")
def api_deploy(model_id: str):
    result = model_service.deploy_to_inference(model_id)
    if not result["ok"]:
        raise HTTPException(500, result["error"])
    return result


@router.post("/{model_id}/rollback")
def api_rollback(model_id: str):
    return api_promote(model_id, status="rollback")
