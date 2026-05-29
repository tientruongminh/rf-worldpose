from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = Field(default="postgresql://rfpose:rfpose@localhost:5432/rfpose", alias="DATABASE_URL")

    # HPC — SSH-based job submission (Eagle PCSS or Helios Cyfronet)
    hpc_login: str = Field(default="eagle.man.poznan.pl", alias="HPC_LOGIN")
    hpc_user: str = Field(default="tiencd1234", alias="HPC_USER")
    hpc_ssh_key: str = Field(default="/opt/rfpose/.ssh/helios_ed25519", alias="HPC_SSH_KEY")
    hpc_account: str = Field(default="", alias="HPC_ACCOUNT")
    hpc_partition: str = Field(default="", alias="HPC_PARTITION")
    hpc_work_dir: str = Field(default="~/pl0501-01/project_data/tien", alias="HPC_WORK_DIR")

    # S3 / MinIO — reachable from both VPS and Helios compute nodes
    s3_bucket: str = Field(default="rfpose", alias="S3_BUCKET")
    s3_endpoint_url: str = Field(default="http://minio:9000", alias="S3_ENDPOINT_URL")
    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")

    @property
    def hpc_ssh_target(self) -> str:
        """user@host for SSH commands."""
        if self.hpc_user:
            return f"{self.hpc_user}@{self.hpc_login}"
        return self.hpc_login


settings = Settings()
