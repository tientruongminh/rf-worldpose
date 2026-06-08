"""Data access for deployments + nodes tables."""
from __future__ import annotations

from rfpose_api.db.connection import connect


def create(*, id: str, name: str, room_id: str, metadata: str | None = None) -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO deployments(id, name, room_id, metadata)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, room_id=EXCLUDED.room_id,
                 metadata=EXCLUDED.metadata, updated_at=now()
               RETURNING *""",
            (id, name, room_id, metadata),
        )
        return cur.fetchone()


def get(deployment_id: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM deployments WHERE id=%s", (deployment_id,))
        return cur.fetchone()


def get_with_nodes(deployment_id: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM deployments WHERE id=%s", (deployment_id,))
        dep = cur.fetchone()
        if not dep:
            return None
        cur.execute(
            "SELECT * FROM nodes WHERE deployment_id=%s ORDER BY id",
            (deployment_id,),
        )
        return {"deployment": dep, "nodes": cur.fetchall()}


def upsert_node(
    *,
    id: str,
    deployment_id: str,
    hardware_revision: str | None = None,
    firmware_version: str | None = None,
    position: str | None = None,
    status: str | None = None,
) -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO nodes(id, deployment_id, hardware_revision, firmware_version, position, status, last_seen_at)
               VALUES (%s, %s, %s, %s, %s, %s, now())
               ON CONFLICT (id) DO UPDATE SET
                 firmware_version=EXCLUDED.firmware_version, hardware_revision=EXCLUDED.hardware_revision,
                 position=EXCLUDED.position, status=EXCLUDED.status, last_seen_at=now(), updated_at=now()
               RETURNING *""",
            (id, deployment_id, hardware_revision, firmware_version, position, status),
        )
        return cur.fetchone()
