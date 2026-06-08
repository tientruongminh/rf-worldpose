"""RF-WorldPose Inference Service

Subscribes to NATS for real-time CSI packets from the Rust Gateway,
runs ONNX model inference, and publishes results back to NATS.
Also exposes a REST API for manual/portal inference and model management.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import nats
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("rfpose.inference")

NATS_URL = os.environ.get("NATS_URL", "nats://nats:4222")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/rfpose/models/production"))
DEPLOYMENT_ID = os.environ.get("RFPOSE_DEPLOYMENT_ID", "room01")
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "30"))
N_NODES = int(os.environ.get("N_NODES", "9"))
S3_BUCKET = os.environ.get("S3_BUCKET", "rfpose")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "http://minio:9000")

ACTION_LABELS = {
    0: "stand", 1: "walk", 2: "run", 3: "jump", 4: "sit_down",
    5: "stand_up", 6: "bend", 7: "fall", 8: "wave", 9: "clap",
    10: "throw", 11: "catch", 12: "kick", 13: "punch", 14: "push", 15: "pull",
}

app = FastAPI(title="RF-WorldPose Inference Service", version="0.2.0")

_session: ort.InferenceSession | None = None
_nc: nats.NATS | None = None
_buffer: dict[int, deque] = {}
_recent_predictions: deque = deque(maxlen=100)
_stats = {"packets_received": 0, "inferences_run": 0, "last_prediction": None}


# ── Model Management ────────────────────────────────────────

def load_model() -> ort.InferenceSession | None:
    global _session
    model_path = MODEL_DIR / "model.onnx"
    if not model_path.exists():
        log.warning("No model at %s", model_path)
        return None
    try:
        _session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = _session.get_inputs()[0]
        log.info("Loaded model: %s input=%s shape=%s", model_path, inp.name, inp.shape)
        return _session
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        return None


def get_session() -> ort.InferenceSession | None:
    global _session
    if _session is None:
        load_model()
    return _session


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_inference(csi: np.ndarray) -> dict:
    """Run inference on a CSI window. Returns prediction dict."""
    session = get_session()
    if session is None:
        return {"error": "no model loaded"}

    if csi.ndim == 3:
        csi = np.expand_dims(csi, 0)
    csi = csi.astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: csi})

    result: dict = {"timestamp": time.time(), "model": "model.onnx"}

    logits = outputs[0]
    if logits.ndim >= 2:
        probs = softmax(logits[0])
        action_id = int(np.argmax(probs))
        result["action_id"] = action_id
        result["action"] = ACTION_LABELS.get(action_id, f"class_{action_id}")
        result["confidence"] = float(probs[action_id])
        result["probabilities"] = {str(i): round(float(p), 4) for i, p in enumerate(probs)}

    if len(outputs) > 1 and outputs[1].ndim >= 2:
        result["pose_2d"] = outputs[1][0].tolist()

    return result


# ── NATS Subscriber ──────────────────────────────────────────

async def on_csi_message(msg):
    """Handle incoming CSI packet from Gateway via NATS."""
    global _stats
    try:
        pkt = json.loads(msg.data.decode())
        node_id = pkt.get("node_id", 0)
        amplitude = pkt.get("amplitude", [])

        if node_id not in _buffer:
            _buffer[node_id] = deque(maxlen=WINDOW_SIZE)
        _buffer[node_id].append(amplitude)
        _stats["packets_received"] += 1

        active_nodes = [nid for nid, buf in _buffer.items() if len(buf) >= WINDOW_SIZE]
        if len(active_nodes) >= min(N_NODES, len(_buffer)) and len(active_nodes) > 0:
            window = []
            for nid in sorted(active_nodes)[:N_NODES]:
                window.append(list(_buffer[nid]))

            csi = np.array(window, dtype=np.float32)
            result = run_inference(csi)

            if "error" not in result:
                _stats["inferences_run"] += 1
                _stats["last_prediction"] = result
                _recent_predictions.append(result)

                if _nc:
                    await _nc.publish(
                        f"inference.result.{DEPLOYMENT_ID}",
                        json.dumps(result).encode(),
                    )
                    log.debug("Published prediction: action=%s conf=%.2f",
                              result.get("action"), result.get("confidence", 0))

                for nid in active_nodes:
                    _buffer[nid].clear()

    except Exception as exc:
        log.warning("Error processing CSI message: %s", exc)


async def nats_subscriber():
    """Connect to NATS and subscribe to CSI topics."""
    global _nc
    retry_delay = 5
    while True:
        try:
            _nc = await nats.connect(NATS_URL)
            log.info("Connected to NATS at %s", NATS_URL)

            subject = f"csi.raw.{DEPLOYMENT_ID}.>"
            await _nc.subscribe(subject, cb=on_csi_message)
            log.info("Subscribed to %s", subject)

            while _nc.is_connected:
                await asyncio.sleep(1)

        except Exception as exc:
            log.warning("NATS connection error: %s (retry in %ds)", exc, retry_delay)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)


# ── FastAPI Lifecycle ────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_model()
    asyncio.create_task(nats_subscriber())
    log.info("Inference service started (deployment=%s, window=%d, nodes=%d)",
             DEPLOYMENT_ID, WINDOW_SIZE, N_NODES)


# ── REST Endpoints ───────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _session is not None,
        "nats_connected": _nc is not None and _nc.is_connected if _nc else False,
    }


@app.get("/status")
def status():
    model_path = MODEL_DIR / "model.onnx"
    result = {
        "model_dir": str(MODEL_DIR),
        "model_exists": model_path.exists(),
        "model_loaded": _session is not None,
        "nats_url": NATS_URL,
        "nats_connected": _nc is not None and _nc.is_connected if _nc else False,
        "deployment_id": DEPLOYMENT_ID,
        "window_size": WINDOW_SIZE,
        "n_nodes": N_NODES,
        "stats": _stats,
        "buffer_nodes": {nid: len(buf) for nid, buf in _buffer.items()},
    }
    if _session:
        result["input_names"] = [i.name for i in _session.get_inputs()]
        result["input_shapes"] = [i.shape for i in _session.get_inputs()]
        result["output_names"] = [o.name for o in _session.get_outputs()]
    available = list(MODEL_DIR.glob("*.onnx")) if MODEL_DIR.exists() else []
    result["available_models"] = [f.name for f in available]
    return result


class CSIInput(BaseModel):
    csi: list[list[list[float]]]


class PredictResponse(BaseModel):
    action: str | None = None
    action_id: int | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    pose_2d: list[list[float]] | None = None
    model: str = ""
    error: str | None = None


@app.post("/predict", response_model=PredictResponse)
def predict_manual(payload: CSIInput):
    """Manual inference for testing / portal. Real-time inference flows through NATS."""
    csi = np.array(payload.csi, dtype=np.float32)
    result = run_inference(csi)
    if "error" in result:
        raise HTTPException(503, result["error"])
    return PredictResponse(**{k: v for k, v in result.items() if k in PredictResponse.model_fields})


@app.post("/reload")
def reload():
    """Hot-reload model from disk."""
    global _session
    _session = None
    session = load_model()
    if session:
        return {"status": "reloaded", "model": "model.onnx"}
    raise HTTPException(503, "No model found to reload.")


@app.get("/predictions/recent")
def recent_predictions(limit: int = 20):
    """Last N predictions from the NATS real-time pipeline."""
    items = list(_recent_predictions)[-limit:]
    return {"predictions": items, "total": len(_recent_predictions)}
