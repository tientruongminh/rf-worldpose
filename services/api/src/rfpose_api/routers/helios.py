from fastapi import APIRouter, HTTPException
from rfpose_api.db.connection import connect
from rfpose_api.config import settings
import sys, logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[5] / "helios_runner"))
from rfpose_helios.submit import HeliosJobSpec, submit_training_job, submit_script, test_connection, list_remote_scripts
from rfpose_helios.status import slurm_status
from rfpose_helios.cancel import cancel_job

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hpc", tags=["hpc"])

SLURM_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}

_login = settings.hpc_ssh_target
_ssh_key = settings.hpc_ssh_key
_work_dir = settings.hpc_work_dir


@router.get("/connection-test")
def connection_test():
    return test_connection(_login, ssh_key=_ssh_key)


@router.get("/remote-scripts")
def remote_scripts():
    """List .sh/.sbatch scripts available on HPC work dir."""
    scripts = list_remote_scripts(_login, ssh_key=_ssh_key, remote_dir=_work_dir)
    return {"work_dir": _work_dir, "scripts": scripts}


@router.post("/submit-script")
def submit_remote_script(script_name: str):
    """Submit an existing script on HPC directly."""
    try:
        slurm_id = submit_script(
            login=_login, ssh_key=_ssh_key,
            remote_dir=_work_dir, script_name=script_name,
        )
    except Exception as exc:
        raise HTTPException(502, f"submit failed: {exc}")
    return {"slurm_job_id": slurm_id, "script": script_name}


@router.post("/training-jobs/{job_id}/submit")
def submit_job(job_id: str, dry_run: bool = True):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        job = cur.fetchone()
        if not job:
            raise HTTPException(404, "training job not found")

        spec = HeliosJobSpec(
            job_id=job_id,
            dataset_version=job["dataset_version"],
            train_config=job["train_config"],
            account=settings.hpc_account,
            partition=settings.hpc_partition,
            s3_bucket=settings.s3_bucket,
            s3_endpoint_url=settings.s3_endpoint_url,
            mlflow_tracking_uri=settings.mlflow_tracking_uri,
        )
        result = submit_training_job(
            spec, login=_login, ssh_key=_ssh_key,
            remote_dir=_work_dir, dry_run=dry_run,
        )

        if dry_run:
            return {"dry_run": True, "sbatch": result}

        cur.execute(
            """UPDATE training_jobs
               SET status='submitted', slurm_job_id=%s, submitted_at=now(), updated_at=now()
               WHERE id=%s RETURNING *""",
            (result, job_id),
        )
        return cur.fetchone()


@router.post("/training-jobs/{job_id}/refresh-status")
def refresh_job_status(job_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        job = cur.fetchone()
        if not job:
            raise HTTPException(404, "training job not found")
        if not job.get("slurm_job_id"):
            raise HTTPException(400, "job has no slurm_job_id — not yet submitted")

        try:
            raw = slurm_status(_login, job["slurm_job_id"], ssh_key=_ssh_key)
        except Exception as exc:
            log.warning("slurm status check failed for %s: %s", job_id, exc)
            raise HTTPException(502, f"failed to reach Slurm: {exc}")

        parts = raw.split("|") if raw else []
        slurm_state = parts[1] if len(parts) > 1 else "UNKNOWN"
        new_status = _map_slurm_state(slurm_state, job["status"])

        set_parts = ["slurm_state=%s", "status=%s", "updated_at=now()"]
        vals: list = [slurm_state, new_status]

        if new_status == "running" and not job.get("started_at"):
            set_parts.append("started_at=now()")
        if slurm_state in SLURM_TERMINAL and not job.get("finished_at"):
            set_parts.append("finished_at=COALESCE(finished_at, now())")

        vals.append(job_id)
        cur.execute(
            f"UPDATE training_jobs SET {', '.join(set_parts)} WHERE id=%s RETURNING *", vals,
        )
        return cur.fetchone()


@router.post("/training-jobs/{job_id}/cancel")
def cancel_slurm_job(job_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        job = cur.fetchone()
        if not job:
            raise HTTPException(404, "training job not found")
        if not job.get("slurm_job_id"):
            raise HTTPException(400, "job has no slurm_job_id")

        try:
            cancel_job(_login, job["slurm_job_id"], ssh_key=_ssh_key)
        except Exception as exc:
            raise HTTPException(502, f"cancel failed: {exc}")

        cur.execute(
            """UPDATE training_jobs
               SET status='cancelled', slurm_state='CANCELLED', finished_at=now(), updated_at=now()
               WHERE id=%s RETURNING *""",
            (job_id,),
        )
        return cur.fetchone()


def _map_slurm_state(slurm_state: str, current_status: str) -> str:
    mapping = {
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
    return mapping.get(slurm_state, current_status)
