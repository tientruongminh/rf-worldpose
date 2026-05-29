"""Shared schemas and re-exports for backward compatibility."""
from pydantic import BaseModel

from rfpose_api.schemas.deployments import DeploymentCreate, DeploymentOut, NodeUpsert
from rfpose_api.schemas.sessions import RecordingSessionCreate
from rfpose_api.schemas.datasets import DatasetVersionCreate
from rfpose_api.schemas.training import TrainingJobCreate, TrainingJobOut, TrainingJobSubmit
from rfpose_api.schemas.models import ModelVersionCreate


class ApiMessage(BaseModel):
    status: str = "ok"


__all__ = [
    "ApiMessage",
    "DeploymentCreate", "DeploymentOut", "NodeUpsert",
    "RecordingSessionCreate",
    "DatasetVersionCreate",
    "TrainingJobCreate", "TrainingJobOut", "TrainingJobSubmit",
    "ModelVersionCreate",
]
