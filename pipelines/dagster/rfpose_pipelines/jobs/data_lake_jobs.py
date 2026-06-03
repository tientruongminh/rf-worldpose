from __future__ import annotations

from dagster import AssetSelection, define_asset_job


data_lake_job = define_asset_job(
    name="data_lake_job",
    selection=AssetSelection.all(),
    description="Build bronze, silver, and gold multitask data lake artifacts.",
)
