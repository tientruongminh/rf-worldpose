from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


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
