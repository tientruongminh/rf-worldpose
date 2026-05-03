from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = Field(default="postgresql://rfpose:rfpose@localhost:5432/rfpose", alias="DATABASE_URL")
    helios_login: str = Field(default="login01.helios.cyfronet.pl", alias="HELIOS_LOGIN")
    helios_account: str = Field(default="CHANGE_ME-gpu-gh200", alias="HELIOS_ACCOUNT")
    helios_partition: str = Field(default="plgrid-gpu-gh200", alias="HELIOS_PARTITION")
    s3_bucket: str = Field(default="rfpose", alias="S3_BUCKET")
    s3_endpoint_url: str = Field(default="http://minio:9000", alias="S3_ENDPOINT_URL")
    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")

settings = Settings()
