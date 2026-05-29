from datetime import datetime
from pydantic import BaseModel


class TrainingJobCreate(BaseModel):
    id: str
    dataset_version: str
    train_config: str
    backend: str = "eagle-slurm"
    submitted_by: str | None = None


class TrainingJobOut(TrainingJobCreate):
    status: str
    slurm_job_id: str | None = None
    slurm_state: str | None = None
    artifact_uri: str | None = None
    eval_report_uri: str | None = None
    logs_uri: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None


class TrainingJobSubmit(BaseModel):
    dataset_version: str
    train_config: str
    submitted_by: str
    backend: str = "eagle-slurm"
    dry_run: bool = False
