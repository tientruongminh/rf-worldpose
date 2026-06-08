from pydantic import Field
from pydantic_settings import BaseSettings


class PortalSettings(BaseSettings):
    api_base_url: str = Field(default="http://api:8080", alias="API_BASE_URL")
    inference_url: str = Field(default="http://inference:8081", alias="INFERENCE_URL")
    mlflow_url: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")
    dagster_url: str = Field(default="http://dagster:3001", alias="DAGSTER_URL")


settings = PortalSettings()
