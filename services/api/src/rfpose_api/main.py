import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rfpose_api.schemas.common import ApiMessage
from rfpose_api.routers import deployments, sessions, datasets, training, models, hpc, configs
from rfpose_api.tasks.job_poller import run_status_poller

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    poller = asyncio.create_task(run_status_poller())
    yield
    poller.cancel()
    try:
        await poller
    except asyncio.CancelledError:
        pass


app = FastAPI(title="RF-WorldPose API", version="0.1.0", lifespan=lifespan)

app.include_router(deployments.router)
app.include_router(sessions.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(hpc.router)
app.include_router(configs.router)


@app.get("/health", response_model=ApiMessage)
def health() -> ApiMessage:
    return ApiMessage()
