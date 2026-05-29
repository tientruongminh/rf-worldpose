"""Web portal routes — serves Jinja2 HTML for the Job Portal UI."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from rfpose_api.config import settings
from rfpose_api.db.connection import connect

sys.path.append(str(Path(__file__).resolve().parents[5] / "helios_runner"))
from rfpose_helios.submit import HeliosJobSpec, submit_training_job, submit_script, test_connection, list_remote_scripts
from rfpose_helios.status import slurm_status
from rfpose_helios.cancel import cancel_job as _cancel_job

log = logging.getLogger(__name__)

router = APIRouter(prefix="/portal", tags=["portal"], default_response_class=HTMLResponse)
_templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

_login = settings.hpc_ssh_target
_ssh_key = settings.hpc_ssh_key
_work_dir = settings.hpc_work_dir

STATUS_MAP = {
    "PENDING": "submitted", "RUNNING": "running", "COMPLETING": "running",
    "COMPLETED": "completed", "FAILED": "failed", "CANCELLED": "cancelled",
    "TIMEOUT": "failed", "OUT_OF_MEMORY": "failed", "NODE_FAIL": "failed",
}


def _render(request: Request, name: str, ctx: dict):
    return _templates.TemplateResponse(request=request, name=name, context=ctx)


# ── Dashboard ───────────────────────────────────────────────

@router.get("")
def dashboard(request: Request, status: str | None = None):
    with connect() as conn, conn.cursor() as cur:
        if status:
            cur.execute(
                "SELECT * FROM training_jobs WHERE status=%s ORDER BY created_at DESC LIMIT 100",
                (status,),
            )
        else:
            cur.execute("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT 100")
        jobs = cur.fetchall()

        cur.execute(
            """SELECT
                 count(*) as total,
                 count(*) FILTER (WHERE status='running') as running,
                 count(*) FILTER (WHERE status='submitted') as submitted,
                 count(*) FILTER (WHERE status='completed') as completed,
                 count(*) FILTER (WHERE status='failed') as failed
               FROM training_jobs"""
        )
        stats = cur.fetchone()

    return _render(request, "dashboard.html", {
        "jobs": jobs, "stats": stats, "filter_status": status,
    })


# ── Quick Submit (existing HPC script) ─────────────────────

@router.get("/quick-submit")
def quick_submit_form(request: Request):
    scripts = list_remote_scripts(_login, ssh_key=_ssh_key, remote_dir=_work_dir)
    return _render(request, "quick_submit.html", {
        "scripts": scripts, "work_dir": _work_dir, "result": None, "error": None,
    })


@router.post("/quick-submit")
def quick_submit(request: Request, script_name: str = Form(...), submitted_by: str = Form("")):
    scripts = list_remote_scripts(_login, ssh_key=_ssh_key, remote_dir=_work_dir)
    try:
        slurm_id = submit_script(login=_login, ssh_key=_ssh_key, remote_dir=_work_dir, script_name=script_name)

        with connect() as conn, conn.cursor() as cur:
            job_id = f"quick-{slurm_id}"
            cur.execute(
                """INSERT INTO training_jobs(id, dataset_version, train_config, backend, submitted_by, status, slurm_job_id, submitted_at)
                   VALUES (%s, 'n/a', %s, 'eagle-slurm', %s, 'submitted', %s, now())
                   ON CONFLICT (id) DO NOTHING
                   RETURNING *""",
                (job_id, script_name, submitted_by or "portal", slurm_id),
            )

        return _render(request, "quick_submit.html", {
            "scripts": scripts, "work_dir": _work_dir,
            "result": {"slurm_job_id": slurm_id, "script": script_name, "job_id": job_id},
            "error": None,
        })
    except Exception as exc:
        return _render(request, "quick_submit.html", {
            "scripts": scripts, "work_dir": _work_dir, "result": None, "error": str(exc),
        })


# ── Full Submit (generate sbatch) ──────────────────────────

@router.get("/submit")
def submit_form(request: Request):
    datasets = _list_datasets()
    return _render(request, "submit.html", {
        "datasets": datasets, "form": None, "error": None, "dry_run_result": None,
    })


@router.post("/submit")
def submit_job(
    request: Request,
    job_id: str = Form(...),
    submitted_by: str = Form(...),
    dataset_version: str = Form(""),
    dataset_version_manual: str = Form(""),
    train_config: str = Form(...),
    backend: str = Form("eagle-slurm"),
    auto_submit: str = Form(""),
    dry_run: str = Form(""),
):
    ds = dataset_version_manual.strip() or dataset_version.strip()
    form_data = {
        "job_id": job_id, "submitted_by": submitted_by, "dataset_version": ds,
        "dataset_version_manual": dataset_version_manual,
        "train_config": train_config, "auto_submit": auto_submit,
    }

    if not ds:
        return _render(request, "submit.html", {
            "datasets": _list_datasets(), "form": form_data,
            "error": "Please select or type a dataset version.", "dry_run_result": None,
        })

    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO training_jobs(id, dataset_version, train_config, backend, submitted_by, status)
                   VALUES (%s, %s, %s, %s, %s, 'created')
                   RETURNING *""",
                (job_id, ds, train_config, backend, submitted_by),
            )
            job = cur.fetchone()
    except Exception as exc:
        log.warning("create job failed: %s", exc)
        return _render(request, "submit.html", {
            "datasets": _list_datasets(), "form": form_data,
            "error": f"Failed to create job: {exc}", "dry_run_result": None,
        })

    if auto_submit or dry_run:
        return _try_submit(request, job, dry_run=bool(dry_run))

    return RedirectResponse(f"/portal/jobs/{job_id}", status_code=303)


# ── Job Detail ──────────────────────────────────────────────

@router.get("/jobs/{job_id}")
def job_detail(request: Request, job_id: str):
    job = _get_job(job_id)
    if not job:
        return RedirectResponse("/portal", status_code=303)
    return _render(request, "job_detail.html", {"job": job, "message": None})


@router.post("/jobs/{job_id}/refresh")
def refresh_status_web(request: Request, job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("slurm_job_id"):
        return _job_fragment(request, job, message="No Slurm job to refresh", message_type="error")

    try:
        raw = slurm_status(_login, job["slurm_job_id"], ssh_key=_ssh_key)
        parts = raw.split("|") if raw else []
        slurm_state = parts[1] if len(parts) > 1 else "UNKNOWN"
        new_status = STATUS_MAP.get(slurm_state, job["status"])

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE training_jobs
                   SET slurm_state=%s, status=%s, updated_at=now()
                   WHERE id=%s RETURNING *""",
                (slurm_state, new_status, job_id),
            )
            job = cur.fetchone()

        return _job_fragment(request, job, message=f"Status refreshed: {slurm_state}")
    except Exception as exc:
        return _job_fragment(request, job, message=f"Refresh failed: {exc}", message_type="error")


@router.post("/jobs/{job_id}/cancel")
def cancel_web(request: Request, job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("slurm_job_id"):
        return _job_fragment(request, job, message="No Slurm job to cancel", message_type="error")

    try:
        _cancel_job(_login, job["slurm_job_id"], ssh_key=_ssh_key)
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE training_jobs
                   SET status='cancelled', slurm_state='CANCELLED', finished_at=now(), updated_at=now()
                   WHERE id=%s RETURNING *""",
                (job_id,),
            )
            job = cur.fetchone()
        return _job_fragment(request, job, message="Job cancelled successfully.")
    except Exception as exc:
        return _job_fragment(request, job, message=f"Cancel failed: {exc}", message_type="error")


@router.post("/jobs/{job_id}/submit")
def submit_existing_web(request: Request, job_id: str):
    job = _get_job(job_id)
    if not job:
        return RedirectResponse("/portal", status_code=303)
    return _try_submit(request, job, dry_run=False)


# ── HPC Connection ──────────────────────────────────────────

@router.get("/connection")
def connection_page(request: Request):
    return _render(request, "connection.html", {"config": settings})


@router.get("/connection-test")
def connection_test_web(request: Request):
    result = test_connection(_login, ssh_key=_ssh_key)

    if result["ok"]:
        html = f"""
        <div class="flash flash-success">Connected to <strong>{result['hostname']}</strong></div>
        <p class="text-sm mt-2" style="color:var(--c-muted);">Queue preview:</p>
        <pre class="sbatch" style="margin-top:0.5rem; font-size:0.75rem;">{result.get('queue_preview') or '(no jobs in queue)'}</pre>
        """
    else:
        html = f'<div class="flash flash-error">Connection failed: {result["error"]}</div>'

    return HTMLResponse(html)


# ── Helpers ─────────────────────────────────────────────────

def _get_job(job_id: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_jobs WHERE id=%s", (job_id,))
        return cur.fetchone()


def _list_datasets():
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM dataset_versions ORDER BY created_at DESC LIMIT 50")
            return cur.fetchall()
    except Exception:
        return []


def _job_fragment(request: Request, job, *, message=None, message_type="success"):
    return _render(request, "job_info_fragment.html", {
        "job": job, "message": message, "message_type": message_type,
    })


def _try_submit(request: Request, job, *, dry_run: bool):
    spec = HeliosJobSpec(
        job_id=job["id"],
        dataset_version=job["dataset_version"],
        train_config=job["train_config"],
        account=settings.hpc_account,
        partition=settings.hpc_partition,
        s3_bucket=settings.s3_bucket,
        s3_endpoint_url=settings.s3_endpoint_url,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
    )

    try:
        result = submit_training_job(
            spec, login=_login, ssh_key=_ssh_key,
            remote_dir=_work_dir, dry_run=dry_run,
        )
    except Exception as exc:
        return _render(request, "job_detail.html", {
            "job": job, "message": f"Submit failed: {exc}", "message_type": "error",
        })

    if dry_run:
        return _render(request, "submit.html", {
            "datasets": _list_datasets(), "form": None, "error": None, "dry_run_result": result,
        })

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE training_jobs
               SET status='submitted', slurm_job_id=%s, submitted_at=now(), updated_at=now()
               WHERE id=%s RETURNING *""",
            (result, job["id"]),
        )
        job = cur.fetchone()

    return RedirectResponse(f"/portal/jobs/{job['id']}", status_code=303)
