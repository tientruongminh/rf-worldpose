from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = Field(default="postgresql://rfpose:rfpose@localhost:5432/rfpose", alias="DATABASE_URL")

    # HPC
    hpc_login: str = Field(default="", alias="HPC_LOGIN")
    hpc_user: str = Field(default="", alias="HPC_USER")
    hpc_ssh_key: str = Field(default="", alias="HPC_SSH_KEY")
    hpc_account: str = Field(default="", alias="HPC_ACCOUNT")
    hpc_partition: str = Field(default="", alias="HPC_PARTITION")
    hpc_work_dir: str = Field(default="~/rfpose-jobs", alias="HPC_WORK_DIR")

    # S3 / MinIO
    s3_bucket: str = Field(default="rfpose", alias="S3_BUCKET")
    s3_endpoint_url: str = Field(default="http://minio:9000", alias="S3_ENDPOINT_URL")
    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")

    # Inference service
    inference_container: str = Field(default="docker-compose-inference-1", alias="INFERENCE_CONTAINER")
    inference_url: str = Field(default="http://inference:8081", alias="INFERENCE_URL")

    # Background poller
    poll_interval_seconds: int = Field(default=120, alias="POLL_INTERVAL_SECONDS")

    @property
    def hpc_ssh_target(self) -> str:
        if self.hpc_user:
            return f"{self.hpc_user}@{self.hpc_login}"
        return self.hpc_login


settings = Settings()
