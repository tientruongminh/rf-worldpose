"""HTTP client to call the RF-WorldPose REST API.

Portal never touches DB directly — all data flows through the API service.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any

from rfpose_portal.config import settings

log = logging.getLogger(__name__)

_BASE = settings.api_base_url


def _request(method: str, path: str, body: dict | None = None, timeout: int = 30) -> Any:
    url = f"{_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        log.warning("%s %s → %d", method, path, exc.code)
        try:
            detail = json.loads(exc.read())
        except Exception:
            detail = {"detail": str(exc)}
        return {"_error": True, "status": exc.code, **detail}
    except Exception as exc:
        log.warning("%s %s failed: %s", method, path, exc)
        return {"_error": True, "detail": str(exc)}


def get(path: str, **kwargs) -> Any:
    return _request("GET", path, **kwargs)


def post(path: str, body: dict | None = None, **kwargs) -> Any:
    return _request("POST", path, body=body, **kwargs)


# ── Training Jobs ────────────────────────────────────────────

def list_jobs(status: str | None = None, limit: int = 100) -> list[dict]:
    qs = f"?limit={limit}"
    if status:
        qs += f"&status={status}"
    data = get(f"/api/v1/training-jobs{qs}")
    return data.get("items", []) if not data.get("_error") else []


def get_job(job_id: str) -> dict | None:
    data = get(f"/api/v1/training-jobs/{job_id}")
    return None if data.get("_error") else data


def create_job(*, job_id: str, dataset_version: str, train_config: str,
               backend: str = "eagle-slurm", submitted_by: str = "portal") -> dict:
    return post("/api/v1/training-jobs", {
        "id": job_id, "dataset_version": dataset_version,
        "train_config": train_config, "backend": backend, "submitted_by": submitted_by,
    })


# ── HPC ──────────────────────────────────────────────────────

def hpc_connection_test() -> dict:
    return get("/api/v1/hpc/connection-test")


def hpc_list_scripts() -> dict:
    return get("/api/v1/hpc/remote-scripts")


def hpc_submit_script(script_name: str) -> dict:
    return post(f"/api/v1/hpc/submit-script?script_name={script_name}")


def hpc_submit_job(job_id: str, dry_run: bool = False) -> dict:
    return post(f"/api/v1/hpc/training-jobs/{job_id}/submit?dry_run={str(dry_run).lower()}")


def hpc_refresh(job_id: str) -> dict:
    return post(f"/api/v1/hpc/training-jobs/{job_id}/refresh-status")


def hpc_cancel(job_id: str) -> dict:
    return post(f"/api/v1/hpc/training-jobs/{job_id}/cancel")


# ── Models ───────────────────────────────────────────────────

def list_models() -> list[dict]:
    data = get("/api/v1/models")
    return data if isinstance(data, list) else []


def get_model(model_id: str) -> dict | None:
    data = get(f"/api/v1/models/{model_id}")
    return None if data.get("_error") else data


def promote_model(model_id: str) -> dict:
    return post(f"/api/v1/models/{model_id}/promote?status=production")


def deploy_model(model_id: str) -> dict:
    return post(f"/api/v1/models/{model_id}/deploy")


# ── Datasets ─────────────────────────────────────────────────

def list_datasets() -> list[dict]:
    data = get("/api/v1/datasets")
    return data if isinstance(data, list) else []
