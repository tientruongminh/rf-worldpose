"""RF-WorldPose Portal — Frontend service.

Serves HTML UI, calls the REST API for all data.
Does NOT touch the database directly.
"""
import logging
from fastapi import FastAPI
from rfpose_portal.routers import portal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="RF-WorldPose Portal", version="0.1.0")
app.include_router(portal.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "portal"}
