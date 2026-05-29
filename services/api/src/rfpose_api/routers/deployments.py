"""REST API for Deployments — thin router."""
from fastapi import APIRouter, HTTPException
from rfpose_api.schemas.common import DeploymentCreate, NodeUpsert
from rfpose_api.repositories import deployments as dep_repo

router = APIRouter(prefix="/api/v1/deployments", tags=["deployments"])


@router.post("")
def create_deployment(payload: DeploymentCreate):
    return dep_repo.create(
        id=payload.id, name=payload.name,
        room_id=payload.room_id, metadata=payload.metadata,
    )


@router.get("/{deployment_id}/status")
def deployment_status(deployment_id: str):
    result = dep_repo.get_with_nodes(deployment_id)
    if not result:
        raise HTTPException(404, "deployment not found")
    return result


@router.put("/{deployment_id}/nodes/{node_id}")
def upsert_node(deployment_id: str, node_id: str, payload: NodeUpsert):
    if payload.deployment_id != deployment_id or payload.id != node_id:
        raise HTTPException(400, "path ids and payload ids must match")
    return dep_repo.upsert_node(
        id=payload.id, deployment_id=payload.deployment_id,
        hardware_revision=payload.hardware_revision,
        firmware_version=payload.firmware_version,
        position=payload.position, status=payload.status,
    )
