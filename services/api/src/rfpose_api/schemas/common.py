from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

class ApiMessage(BaseModel):
    status: str = "ok"

class DeploymentCreate(BaseModel):
    id: str
    name: str
    room_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class DeploymentOut(DeploymentCreate):
    status: str
    created_at: datetime | None = None

class NodeUpsert(BaseModel):
    id: str
    deployment_id: str
    hardware_revision: str | None = None
    firmware_version: str | None = None
    position: dict[str, Any] = Field(default_factory=dict)
    status: str = "online"

class RecordingSessionCreate(BaseModel):
    id: str
    deployment_id: str
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class DatasetVersionCreate(BaseModel):
    id: str
    source_sessions: list[str] = Field(default_factory=list)
    preprocess_version: str
    teacher_version: str | None = None
    artifact_uri: str
    stats: dict[str, Any] = Field(default_factory=dict)
    quality_report_uri: str | None = None
    created_by: str | None = None

class TrainingJobCreate(BaseModel):
    id: str
    dataset_version: str
    train_config: str
    backend: str = "eagle-slurm"
    submitted_by: str | None = None

class TrainingJobOut(TrainingJobCreate):
    status: str
    slurm_job_id: str | None = None
    artifact_uri: str | None = None
    eval_report_uri: str | None = None
    logs_uri: str | None = None
    error_message: str | None = None

class ModelVersionCreate(BaseModel):
    id: str
    name: str
    dataset_version: str | None = None
    training_job_id: str | None = None
    artifact_uri: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    eval_report_uri: str | None = None
    hash: str | None = None
