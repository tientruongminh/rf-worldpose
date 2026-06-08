from dagster import Definitions, load_assets_from_modules
from .assets import data_lake
from .jobs.training_jobs import auto_train_on_gold
from .sensors.data_sensors import gold_ready_sensor, new_bronze_sensor

defs = Definitions(
    assets=load_assets_from_modules([data_lake]),
    jobs=[auto_train_on_gold],
    sensors=[gold_ready_sensor, new_bronze_sensor],
)
