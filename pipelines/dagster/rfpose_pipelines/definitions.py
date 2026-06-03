from __future__ import annotations

from dagster import Definitions, load_assets_from_modules

from .assets import data_lake
from .jobs import data_lake_job
from .sensors import bronze_data_sensor


defs = Definitions(
    assets=load_assets_from_modules([data_lake]),
    jobs=[data_lake_job],
    sensors=[bronze_data_sensor],
)
