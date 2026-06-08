"""Dagster sensors for detecting new data and triggering pipelines.

- gold_ready_sensor: watches for new Gold datasets in S3 → triggers auto_train_on_gold
- new_bronze_sensor: watches for new Bronze uploads → triggers ETL materialization
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from dagster import (
    RunRequest,
    SensorEvaluationContext,
    sensor,
    DefaultSensorStatus,
    AssetKey,
    asset_sensor,
    EventLogEntry,
)


S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "http://minio:9000")
S3_BUCKET = os.environ.get("S3_BUCKET", "rfpose")


@asset_sensor(
    asset_key=AssetKey("gold_dataset"),
    job_name="auto_train_on_gold",
    default_status=DefaultSensorStatus.STOPPED,
    description="When gold_dataset asset materializes, auto-submit HPC training jobs",
    minimum_interval_seconds=60,
)
def gold_ready_sensor(context: SensorEvaluationContext, asset_event: EventLogEntry):
    """Trigger training when a new Gold dataset is materialized."""
    metadata = asset_event.dagster_event.event_specific_data.materialization.metadata

    dataset_version = "unknown"
    if "dataset_version" in metadata:
        dataset_version = metadata["dataset_version"].value

    context.log.info("Gold dataset materialized: %s", dataset_version)

    yield RunRequest(
        run_key=f"gold-train-{dataset_version}-{datetime.utcnow().isoformat()}",
        run_config={
            "ops": {},
        },
        tags={
            "dataset_version": dataset_version,
            "trigger": "gold_ready_sensor",
        },
    )


@sensor(
    description="Watch MinIO for new Bronze uploads and trigger ETL",
    default_status=DefaultSensorStatus.STOPPED,
    minimum_interval_seconds=300,
)
def new_bronze_sensor(context: SensorEvaluationContext):
    """Poll S3 for new Bronze data and trigger ETL pipeline."""
    try:
        import boto3
    except ImportError:
        context.log.warning("boto3 not installed — cannot poll S3")
        return

    cursor = context.cursor or ""

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    )

    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="bronze/", Delimiter="/")
    except Exception as exc:
        context.log.warning("Failed to list S3 bronze/: %s", exc)
        return

    prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    new_datasets = [p for p in prefixes if p > cursor] if cursor else []

    if not new_datasets:
        return

    latest = max(new_datasets)
    context.update_cursor(latest)

    context.log.info("New Bronze data detected: %s", new_datasets)

    yield RunRequest(
        run_key=f"bronze-etl-{latest.rstrip('/')}",
        tags={"trigger": "new_bronze_sensor", "bronze_prefix": latest},
    )
