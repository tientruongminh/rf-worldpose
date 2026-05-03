from fastapi import APIRouter, HTTPException
from rfpose_api.db.connection import connect
from rfpose_api.schemas.common import DeploymentCreate, NodeUpsert

router = APIRouter(prefix="/api/v1/deployments", tags=["deployments"])

@router.post("")
def create_deployment(payload: DeploymentCreate):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO deployments(id, name, room_id, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, room_id=EXCLUDED.room_id, metadata=EXCLUDED.metadata, updated_at=now()
            RETURNING *
            """,
            (payload.id, payload.name, payload.room_id, payload.metadata),
        )
        return cur.fetchone()

@router.get("/{deployment_id}/status")
def deployment_status(deployment_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM deployments WHERE id=%s", (deployment_id,))
        dep = cur.fetchone()
        if not dep:
            raise HTTPException(404, "deployment not found")
        cur.execute("SELECT * FROM nodes WHERE deployment_id=%s ORDER BY id", (deployment_id,))
        nodes = cur.fetchall()
        return {"deployment": dep, "nodes": nodes}

@router.put("/{deployment_id}/nodes/{node_id}")
def upsert_node(deployment_id: str, node_id: str, payload: NodeUpsert):
    if payload.deployment_id != deployment_id or payload.id != node_id:
        raise HTTPException(400, "path ids and payload ids must match")
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO nodes(id, deployment_id, hardware_revision, firmware_version, position, status, last_seen_at)
            VALUES (%s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (id) DO UPDATE SET
              firmware_version=EXCLUDED.firmware_version,
              hardware_revision=EXCLUDED.hardware_revision,
              position=EXCLUDED.position,
              status=EXCLUDED.status,
              last_seen_at=now(),
              updated_at=now()
            RETURNING *
            """,
            (payload.id, payload.deployment_id, payload.hardware_revision, payload.firmware_version, payload.position, payload.status),
        )
        return cur.fetchone()
