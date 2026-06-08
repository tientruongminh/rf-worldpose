from contextlib import contextmanager

import psycopg_pool
from psycopg.rows import dict_row
from rfpose_api.config import settings

_pool: psycopg_pool.ConnectionPool | None = None


def _get_pool() -> psycopg_pool.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg_pool.ConnectionPool(
            settings.database_url,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def connect():
    return _get_pool().connection()
