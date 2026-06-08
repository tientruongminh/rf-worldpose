"""Background task that polls Slurm for active job statuses."""
from __future__ import annotations

import asyncio
import logging

from rfpose_api.config import settings

log = logging.getLogger(__name__)


async def run_status_poller():
    """Long-running coroutine — started by the FastAPI lifespan."""
    interval = settings.poll_interval_seconds
    log.info("status poller started (interval=%ds)", interval)
    while True:
        try:
            from rfpose_api.services.hpc import poll_active_jobs
            count = await asyncio.to_thread(poll_active_jobs)
            if count:
                log.info("polled %d active jobs", count)
        except asyncio.CancelledError:
            log.info("status poller stopping")
            break
        except Exception:
            log.exception("status poller iteration failed")
        await asyncio.sleep(interval)
