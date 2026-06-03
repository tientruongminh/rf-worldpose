from __future__ import annotations

from dagster import AssetSelection, define_asset_job
from dagster import multiprocess_executor


data_lake_job = define_asset_job(
    name="data_lake_job",
    selection=AssetSelection.all(),
    description="Build bronze, silver, and gold multitask data lake artifacts.",
    executor_def=multiprocess_executor.configured({"max_concurrent": 4}),
)
