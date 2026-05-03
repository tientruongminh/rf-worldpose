from fastapi import FastAPI
from rfpose_api.schemas.common import ApiMessage
from rfpose_api.routers import deployments, sessions, datasets, training, models, helios

app = FastAPI(title="RF-WorldPose API", version="0.1.0")
app.include_router(deployments.router)
app.include_router(sessions.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(helios.router)

@app.get("/health", response_model=ApiMessage)
def health() -> ApiMessage:
    return ApiMessage()
