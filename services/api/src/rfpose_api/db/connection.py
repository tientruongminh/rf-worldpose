import psycopg
from psycopg.rows import dict_row
from rfpose_api.config import settings

def connect():
    return psycopg.connect(settings.database_url, row_factory=dict_row)
