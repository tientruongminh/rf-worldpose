from dagster import Definitions, load_assets_from_modules
from .assets import data_lake

defs = Definitions(assets=load_assets_from_modules([data_lake]))
