"""Inference API — receive CSI data, return predictions using deployed ONNX model.

numpy and onnxruntime are imported lazily so the API starts even when they
are not installed (e.g. lightweight containers that only serve the portal).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])

MODEL_DIR = Path("/opt/rfpose/models/production")
_session = None


class CSIInput(BaseModel):
    csi: list[list[list[list[float]]]]
    model_name: str = "latest"


class PredictionResult(BaseModel):
    action: str | None = None
    action_id: int | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    pose_2d: list[list[float]] | None = None
    model_file: str = ""


def _get_session():
    """Lazy-load ONNX model."""
    global _session
    if _session is not None:
        return _session

    model_path = MODEL_DIR / "model.onnx"
    if not model_path.exists():
        return None

    try:
        import onnxruntime as ort
        _session = ort.InferenceSession(str(model_path))
        log.info("Loaded ONNX model: %s", model_path)
        return _session
    except ImportError:
        log.warning("onnxruntime not installed — inference unavailable")
        return None
    except Exception as exc:
        log.warning("Failed to load ONNX model: %s", exc)
        return None


@router.get("/status")
def inference_status():
    model_path = MODEL_DIR / "model.onnx"
    session = _get_session()

    np_ok = True
    try:
        import numpy  # noqa: F401
    except ImportError:
        np_ok = False

    result: dict[str, Any] = {
        "model_dir": str(MODEL_DIR),
        "model_exists": model_path.exists(),
        "model_loaded": session is not None,
        "numpy_installed": np_ok,
    }

    if session:
        result["input_names"] = [i.name for i in session.get_inputs()]
        result["input_shapes"] = [i.shape for i in session.get_inputs()]
        result["output_names"] = [o.name for o in session.get_outputs()]

    available = list(MODEL_DIR.glob("*.onnx")) if MODEL_DIR.exists() else []
    result["available_models"] = [f.name for f in available]

    return result


@router.post("/predict", response_model=PredictionResult)
def predict(payload: CSIInput):
    try:
        import numpy as np
    except ImportError:
        raise HTTPException(503, "numpy not installed in this environment.")

    session = _get_session()
    if session is None:
        raise HTTPException(
            503,
            "No model deployed. Go to /portal/models and deploy a model first.",
        )

    try:
        csi_array = np.array(payload.csi, dtype=np.float32)
        if csi_array.ndim == 3:
            csi_array = np.expand_dims(csi_array, 0)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: csi_array})

        result = PredictionResult(model_file="model.onnx")

        logits = outputs[0]
        if logits.ndim >= 2:
            e = np.exp(logits[0] - np.max(logits[0]))
            probs = e / e.sum()
            action_id = int(np.argmax(probs))
            result.action_id = action_id
            result.confidence = float(probs[action_id])
            result.probabilities = {str(i): float(p) for i, p in enumerate(probs)}

        if len(outputs) > 1:
            pose = outputs[1]
            if pose.ndim >= 2:
                result.pose_2d = pose[0].tolist()

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}")


@router.post("/reload")
def reload_model():
    """Force reload the ONNX model from disk."""
    global _session
    _session = None
    session = _get_session()
    if session:
        return {"status": "reloaded", "model": "model.onnx"}
    raise HTTPException(503, "No model found to reload.")
