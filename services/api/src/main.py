from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RF-WorldPose API", version="0.1.0")

class Health(BaseModel):
    status: str = "ok"

@app.get("/health", response_model=Health)
def health() -> Health:
    return Health()

@app.get("/api/v1/deployments/{deployment_id}/status")
def deployment_status(deployment_id: str):
    return {"deployment_id": deployment_id, "status": "stub"}
