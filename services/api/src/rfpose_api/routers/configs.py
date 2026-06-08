"""REST API for Training Configs — thin router."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from rfpose_api.repositories import training_configs as config_repo

router = APIRouter(prefix="/api/v1/configs", tags=["configs"])


class ConfigBody(BaseModel):
    config_id: str
    label: str
    description: str = ""
    script_path: str = ""
    git_repo: str = ""
    git_branch: str = "main"
    dataset_hint: str = ""
    hyperparams: str = ""
    requirements: str = ""
    created_by: str = "api"


@router.get("")
def list_configs():
    return config_repo.list_all()


@router.get("/{config_id}")
def get_config(config_id: str):
    row = config_repo.get(config_id)
    if not row:
        raise HTTPException(404, "config not found")
    return row


@router.post("")
def create_config(body: ConfigBody):
    return config_repo.create(
        config_id=body.config_id, label=body.label, description=body.description,
        script_path=body.script_path, git_repo=body.git_repo, git_branch=body.git_branch,
        dataset_hint=body.dataset_hint, hyperparams=body.hyperparams,
        requirements=body.requirements, created_by=body.created_by,
    )


@router.put("/{config_id}")
def update_config(config_id: str, body: ConfigBody):
    result = config_repo.update(
        config_id, label=body.label, description=body.description,
        script_path=body.script_path, git_repo=body.git_repo, git_branch=body.git_branch,
        dataset_hint=body.dataset_hint, hyperparams=body.hyperparams,
        requirements=body.requirements, created_by=body.created_by,
    )
    if not result:
        raise HTTPException(404, "config not found")
    return result


@router.delete("/{config_id}")
def delete_config(config_id: str):
    config_repo.delete(config_id)
    return {"status": "deleted"}
