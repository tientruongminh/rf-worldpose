from typing import Any
from pydantic import BaseModel, Field


class ModelVersionCreate(BaseModel):
    id: str
    name: str
    dataset_version: str | None = None
    training_job_id: str | None = None
    artifact_uri: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    eval_report_uri: str | None = None
    hash: str | None = None
