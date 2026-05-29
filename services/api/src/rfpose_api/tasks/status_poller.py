"""Background task that polls Slurm for active job statuses."""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from rfpose_api.config import settings
from rfpose_api.db.connection import connect

sys.path.append(str(Path(__file__).resolve().parents[5] / "helios_runner"))

log = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 120
ACTIVE_STATUSES = ("submitted", "running")
SLURM_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}

STATUS_MAP = {
    "PENDING": "submitted",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
}


async def run_status_poller():
    """Long-running coroutine — started by the FastAPI lifespan."""
    log.info("status poller started (interval=%ds)", POLL_INTERVAL_SECONDS)
    while True:
        try:
            await asyncio.to_thread(_poll_once)
        except asyncio.CancelledError:
            log.info("status poller stopping")
            break
        except Exception:
            log.exception("status poller iteration failed")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def _poll_once():
    from rfpose_helios.status import slurm_status

    login = settings.helios_ssh_target
    ssh_key = settings.helios_ssh_key

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, slurm_job_id, status FROM training_jobs WHERE status = ANY(%s) AND slurm_job_id IS NOT NULL",
            (list(ACTIVE_STATUSES),),
        )
        active_jobs = cur.fetchall()

    if not active_jobs:
        return

    log.info("polling %d active jobs", len(active_jobs))

    for job in active_jobs:
        try:
            raw = slurm_status(login, job["slurm_job_id"], ssh_key=ssh_key)
            parts = raw.split("|") if raw else []
            slurm_state = parts[1] if len(parts) > 1 else "UNKNOWN"
            new_status = STATUS_MAP.get(slurm_state, job["status"])

            set_parts = ["slurm_state=%s", "status=%s", "updated_at=now()"]
            vals = [slurm_state, new_status]

            if new_status == "running" and job["status"] != "running":
                set_parts.append("started_at=now()")
            if slurm_state in SLURM_TERMINAL:
                set_parts.append("finished_at=COALESCE(finished_at, now())")

            vals.append(job["id"])
            sql = f"UPDATE training_jobs SET {', '.join(set_parts)} WHERE id=%s"

            with connect() as conn, conn.cursor() as cur:
                cur.execute(sql, vals)

            log.info("job %s: slurm=%s -> status=%s", job["id"], slurm_state, new_status)

        except Exception:
            log.exception("failed to poll job %s", job["id"])
