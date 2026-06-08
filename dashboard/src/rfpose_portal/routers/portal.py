"""Portal web routes — renders Jinja2 HTML, calls REST API for data.

This is the frontend. It never touches the database directly.
All data comes from the API service via HTTP.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from rfpose_portal.config import settings
from rfpose_portal import api_client as api

log = logging.getLogger(__name__)

router = APIRouter(default_response_class=HTMLResponse)
_templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[2].parent / "templates"))


def _render(request: Request, name: str, ctx: dict):
    return _templates.TemplateResponse(request=request, name=name, context=ctx)


def _job_fragment(request: Request, job, *, message=None, message_type="success"):
    return _render(request, "job_info_fragment.html", {"job": job, "message": message, "message_type": message_type})


# ── Dashboard ────────────────────────────────────────────────

@router.get("/")
def dashboard(request: Request, status: str | None = None):
    jobs = api.list_jobs(status=status)
    stats = api.get("/api/v1/training-jobs/stats")
    return _render(request, "dashboard.html", {"jobs": jobs, "stats": stats, "filter_status": status})


# ── Quick Submit ─────────────────────────────────────────────

@router.get("/quick-submit")
def quick_submit_form(request: Request):
    data = api.hpc_list_scripts()
    return _render(request, "quick_submit.html", {
        "scripts": data.get("scripts", []), "work_dir": data.get("work_dir", ""),
        "result": None, "error": None,
    })


@router.post("/quick-submit")
def quick_submit(request: Request, script_name: str = Form(...), submitted_by: str = Form("")):
    scripts_data = api.hpc_list_scripts()
    scripts = scripts_data.get("scripts", [])
    work_dir = scripts_data.get("work_dir", "")

    result = api.hpc_submit_script(script_name)
    if result.get("_error"):
        return _render(request, "quick_submit.html", {
            "scripts": scripts, "work_dir": work_dir, "result": None, "error": result.get("detail", "Submit failed"),
        })

    slurm_id = result.get("slurm_job_id", "")
    job_id = f"quick-{slurm_id}"
    api.create_job(job_id=job_id, dataset_version="n/a", train_config=script_name,
                   backend="eagle-slurm", submitted_by=submitted_by or "portal")

    return _render(request, "quick_submit.html", {
        "scripts": scripts, "work_dir": work_dir,
        "result": {"slurm_job_id": slurm_id, "script": script_name, "job_id": job_id}, "error": None,
    })


# ── Full Submit ──────────────────────────────────────────────

@router.get("/submit")
def submit_form(request: Request):
    configs = api.get("/api/v1/configs")
    configs = configs if isinstance(configs, list) else []
    configs.append({"id": "custom", "label": "Custom config (type below)",
                    "hyperparams": "", "dataset_hint": "", "script_path": "",
                    "git_repo": "", "git_branch": "", "description": ""})
    return _render(request, "submit.html", {
        "datasets": api.list_datasets(), "presets": configs,
        "form": None, "error": None, "dry_run_result": None,
    })


@router.post("/submit")
def submit_job(
    request: Request,
    job_id: str = Form(...),
    submitted_by: str = Form(...),
    dataset_version: str = Form(""),
    dataset_version_manual: str = Form(""),
    train_config: str = Form(...),
    preset_id: str = Form(""),
    backend: str = Form("eagle-slurm"),
    auto_submit: str = Form(""),
    dry_run: str = Form(""),
):
    ds = dataset_version_manual.strip() or dataset_version.strip()
    if not ds:
        configs = api.get("/api/v1/configs")
        configs = configs if isinstance(configs, list) else []
        configs.append({"id": "custom", "label": "Custom config (type below)",
                        "hyperparams": "", "dataset_hint": "", "script_path": "",
                        "git_repo": "", "git_branch": "", "description": ""})
        return _render(request, "submit.html", {
            "datasets": api.list_datasets(), "presets": configs,
            "form": {"job_id": job_id, "submitted_by": submitted_by},
            "error": "Please select or type a dataset version.", "dry_run_result": None,
        })

    result = api.create_job(job_id=job_id, dataset_version=ds, train_config=train_config,
                            backend=backend, submitted_by=submitted_by)
    if result.get("_error"):
        return _render(request, "submit.html", {
            "datasets": api.list_datasets(), "presets": [],
            "form": {"job_id": job_id}, "error": result.get("detail", "Create failed"), "dry_run_result": None,
        })

    if auto_submit or dry_run:
        sub = api.hpc_submit_job(job_id, dry_run=bool(dry_run))
        if sub.get("dry_run"):
            return _render(request, "submit.html", {
                "datasets": api.list_datasets(), "presets": [],
                "form": None, "error": None, "dry_run_result": sub.get("sbatch"),
            })

    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


# ── Job Detail ───────────────────────────────────────────────

@router.get("/jobs/{job_id}")
def job_detail(request: Request, job_id: str):
    job = api.get_job(job_id)
    if not job:
        return RedirectResponse("/", status_code=303)
    return _render(request, "job_detail.html", {
        "job": job, "message": None, "mlflow_url": settings.mlflow_url,
    })


@router.post("/jobs/{job_id}/refresh")
def refresh_web(request: Request, job_id: str):
    result = api.hpc_refresh(job_id)
    if result.get("_error"):
        job = api.get_job(job_id)
        return _job_fragment(request, job, message=result.get("detail", "Refresh failed"), message_type="error")
    return _job_fragment(request, result, message=f"Status refreshed: {result.get('slurm_state', '?')}")


@router.post("/jobs/{job_id}/cancel")
def cancel_web(request: Request, job_id: str):
    result = api.hpc_cancel(job_id)
    if result.get("_error"):
        job = api.get_job(job_id)
        return _job_fragment(request, job, message=result.get("detail", "Cancel failed"), message_type="error")
    return _job_fragment(request, result, message="Job cancelled successfully.")


@router.post("/jobs/{job_id}/submit")
def submit_existing(request: Request, job_id: str):
    result = api.hpc_submit_job(job_id, dry_run=False)
    if result.get("_error"):
        job = api.get_job(job_id)
        return _render(request, "job_detail.html", {
            "job": job, "message": result.get("detail", "Submit failed"), "message_type": "error",
        })
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.get("/jobs/{job_id}/logs")
def job_logs(request: Request, job_id: str):
    # Logs are fetched directly via HPC SSH — still goes through API
    # For now, proxy to API's portal route (will be replaced with proper API endpoint)
    return HTMLResponse("<pre class='sbatch'>Logs: use API /portal/jobs/{job_id}/logs endpoint</pre>")


# ── Inference ────────────────────────────────────────────────

@router.get("/inference")
def inference_page(request: Request):
    return _render(request, "inference.html", {})


# ── Model Registry ───────────────────────────────────────────

@router.get("/models")
def models_list(request: Request):
    models = api.list_models()
    active = next((m for m in models if m.get("status") == "production"), None)
    return _render(request, "models.html", {
        "models": models, "active_model": active, "message": None, "message_type": None,
    })


@router.post("/models/{model_id}/promote")
def promote_web(request: Request, model_id: str):
    api.promote_model(model_id)
    return RedirectResponse("/models", status_code=303)


@router.post("/models/{model_id}/deploy")
def deploy_web(request: Request, model_id: str):
    result = api.deploy_model(model_id)
    models = api.list_models()
    model = api.get_model(model_id)
    return _render(request, "models.html", {
        "models": models, "active_model": model,
        "message": result.get("message") or result.get("detail", ""),
        "message_type": "error" if result.get("_error") else "success",
    })


# ── HPC Connection ──────────────────────────────────────────

@router.get("/connection")
def connection_page(request: Request):
    return _render(request, "connection.html", {"config": settings})


@router.get("/connection-test")
def connection_test(request: Request):
    result = api.hpc_connection_test()
    if result.get("ok"):
        html = f"""
        <div class="flash flash-success">Connected to <strong>{result['hostname']}</strong></div>
        <pre class="sbatch" style="margin-top:0.5rem;font-size:0.75rem;">{result.get('queue_preview') or '(no jobs)'}</pre>
        """
    else:
        html = f'<div class="flash flash-error">Connection failed: {result.get("error", "unknown")}</div>'
    return HTMLResponse(html)


# ── Config Registry ──────────────────────────────────────────

@router.get("/configs")
def configs_list(request: Request):
    configs = api.get("/api/v1/configs")
    return _render(request, "configs.html", {"configs": configs if isinstance(configs, list) else []})


@router.get("/configs/new")
def config_new(request: Request):
    return _render(request, "config_form.html", {
        "datasets": api.list_datasets(), "config": None, "error": None, "editing": False,
    })


@router.post("/configs/new")
def config_create(
    request: Request,
    config_id: str = Form(...), label: str = Form(...), description: str = Form(""),
    script_path: str = Form(""), git_repo: str = Form(""), git_branch: str = Form("main"),
    dataset_hint: str = Form(""), hyperparams: str = Form(""),
    requirements: str = Form(""), created_by: str = Form("system"),
):
    result = api.post("/api/v1/configs", {
        "config_id": config_id, "label": label, "description": description,
        "script_path": script_path, "git_repo": git_repo, "git_branch": git_branch,
        "dataset_hint": dataset_hint, "hyperparams": hyperparams,
        "requirements": requirements, "created_by": created_by,
    })
    if result.get("_error"):
        return _render(request, "config_form.html", {
            "datasets": api.list_datasets(),
            "config": {"id": config_id, "label": label, "description": description,
                       "script_path": script_path, "git_repo": git_repo, "git_branch": git_branch,
                       "dataset_hint": dataset_hint, "hyperparams": hyperparams,
                       "requirements": requirements, "created_by": created_by},
            "error": result.get("detail", "Create failed"), "editing": False,
        })
    return RedirectResponse("/configs", status_code=303)


@router.get("/configs/{config_id}/edit")
def config_edit(request: Request, config_id: str):
    config = api.get(f"/api/v1/configs/{config_id}")
    if not config or config.get("_error"):
        return RedirectResponse("/configs", status_code=303)
    return _render(request, "config_form.html", {
        "datasets": api.list_datasets(), "config": config, "error": None, "editing": True,
    })


@router.post("/configs/{config_id}/delete")
def config_delete(request: Request, config_id: str):
    api._request("DELETE", f"/api/v1/configs/{config_id}")
    return RedirectResponse("/configs", status_code=303)
