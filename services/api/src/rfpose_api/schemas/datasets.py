from typing import Any
from pydantic import BaseModel, Field


class DatasetVersionCreate(BaseModel):
    id: str
    source_sessions: list[str] = Field(default_factory=list)
    preprocess_version: str
    teacher_version: str | None = None
    artifact_uri: str
    stats: dict[str, Any] = Field(default_factory=dict)
    quality_report_uri: str | None = None
    created_by: str | None = None
