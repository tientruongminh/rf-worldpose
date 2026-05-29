from typing import Any
from pydantic import BaseModel, Field


class RecordingSessionCreate(BaseModel):
    id: str
    deployment_id: str
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
