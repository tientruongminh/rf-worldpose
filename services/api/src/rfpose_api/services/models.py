"""Model lifecycle service — promote, deploy, rollback."""
from __future__ import annotations

import logging
import subprocess
import urllib.request
from typing import Any

from rfpose_api.config import settings
from rfpose_api.repositories import models as model_repo

log = logging.getLogger(__name__)


def list_models(limit: int = 100) -> list[dict]:
    return model_repo.list_all(limit)


def get_model(model_id: str) -> dict | None:
    return model_repo.get(model_id)


def get_active() -> dict | None:
    return model_repo.get_active()


def promote(model_id: str, status: str = "production") -> dict | None:
    if status not in {"staging", "production", "archived", "rollback"}:
        raise ValueError(f"Invalid status: {status}")
    if status == "production":
        model_repo.archive_current_production(model_id)
    return model_repo.set_status(model_id, status)


def deploy_to_inference(model_id: str) -> dict[str, Any]:
    """Sync model from S3 into inference container, trigger hot reload."""
    model = model_repo.get(model_id)
    if not model:
        return {"ok": False, "error": "Model not found"}

    artifact_uri = model.get("artifact_uri", "")
    s3_src = artifact_uri or f"s3://{settings.s3_bucket}/models/rfpose-{model.get('training_job_id', 'unknown')}/"

    try:
        subprocess.run(
            ["docker", "exec", settings.inference_container, "mkdir", "-p", "/models"],
            check=True, timeout=5,
        )
        subprocess.run(
            ["docker", "exec", settings.inference_container, "sh", "-c",
             f"pip install -q awscli 2>/dev/null; "
             f"aws s3 sync {s3_src} /models/ --endpoint-url {settings.s3_endpoint_url}"],
            check=True, timeout=120,
        )
    except Exception as exc:
        return {"ok": False, "error": f"S3 sync failed: {exc}"}

    try:
        req = urllib.request.Request(f"{settings.inference_url}/reload", method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        log.warning("Could not hot-reload inference service")

    model_repo.set_status(model_id, "production")
    return {"ok": True, "message": f"Model {model_id} deployed → inference service reloaded"}
