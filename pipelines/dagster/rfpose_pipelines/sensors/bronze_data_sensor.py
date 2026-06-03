from __future__ import annotations

import os
from pathlib import Path

from dagster import RunRequest, SensorEvaluationContext, SkipReason, sensor

from rfpose_pipelines.etl.bronze_to_silver import (
    is_s3_uri,
    make_s3_client,
    parse_s3_uri,
)
from rfpose_pipelines.jobs import data_lake_job


def _watched_suffixes() -> set[str]:
    value = os.getenv("RFPOSE_SENSOR_EXTENSIONS", ".json,.mat,.npy,.csv,.dat")
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def _latest_bronze_change(bronze_root: Path) -> tuple[int, str | None, int]:
    suffixes = _watched_suffixes()
    latest_mtime_ns = 0
    latest_path = None
    file_count = 0

    for path in bronze_root.rglob("*"):
        if not path.is_file():
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue

        file_count += 1
        mtime_ns = path.stat().st_mtime_ns
        if mtime_ns > latest_mtime_ns:
            latest_mtime_ns = mtime_ns
            latest_path = str(path)

    return latest_mtime_ns, latest_path, file_count


def _latest_s3_bronze_change(bronze_uri: str) -> tuple[int, str | None, int]:
    suffixes = _watched_suffixes()
    bucket, prefix = parse_s3_uri(bronze_uri)
    client = make_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    page_kwargs = {"Bucket": bucket}
    if prefix:
        page_kwargs["Prefix"] = f"{prefix}/"

    latest_mtime_ns = 0
    latest_uri = None
    file_count = 0

    for page in paginator.paginate(**page_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if suffixes and Path(key).suffix.lower() not in suffixes:
                continue

            file_count += 1
            mtime_ns = int(obj["LastModified"].timestamp() * 1_000_000_000)
            if mtime_ns > latest_mtime_ns:
                latest_mtime_ns = mtime_ns
                latest_uri = f"s3://{bucket}/{key}"

    return latest_mtime_ns, latest_uri, file_count


@sensor(
    name="bronze_data_sensor",
    job=data_lake_job,
    minimum_interval_seconds=60,
    description="Materialize the data lake when bronze dataset files are added or changed.",
)
def bronze_data_sensor(context: SensorEvaluationContext):
    bronze_root_value = os.getenv(
        "RFPOSE_BRONZE_ROOT",
        f"s3://{os.getenv('S3_BUCKET', 'rfpose')}/bronze",
    )

    if is_s3_uri(bronze_root_value):
        latest_mtime_ns, latest_path, file_count = _latest_s3_bronze_change(
            bronze_root_value
        )
    else:
        bronze_root = Path(bronze_root_value)
        if not bronze_root.exists():
            return SkipReason(f"Bronze root does not exist: {bronze_root}")

        latest_mtime_ns, latest_path, file_count = _latest_bronze_change(bronze_root)

    if file_count == 0:
        return SkipReason(f"No watched bronze files found under: {bronze_root_value}")

    previous_mtime_ns = int(context.cursor or "0")
    if latest_mtime_ns <= previous_mtime_ns:
        return SkipReason("No new bronze data changes.")

    context.update_cursor(str(latest_mtime_ns))
    return RunRequest(
        run_key=f"bronze-{latest_mtime_ns}",
        tags={
            "rfpose/bronze_root": bronze_root_value,
            "rfpose/latest_file": latest_path or "",
            "rfpose/file_count": str(file_count),
        },
    )
