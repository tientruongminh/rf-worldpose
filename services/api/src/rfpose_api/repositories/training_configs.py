"""Data access for training_configs table."""
from __future__ import annotations
from rfpose_api.db.connection import connect


def get(config_id: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_configs WHERE id=%s", (config_id,))
        return cur.fetchone()


def list_all() -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM training_configs ORDER BY created_at ASC")
        return cur.fetchall()


def create(*, config_id: str, label: str, description: str, script_path: str,
           git_repo: str, git_branch: str, dataset_hint: str,
           hyperparams: str, requirements: str, created_by: str) -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO training_configs
               (id, label, description, script_path, git_repo, git_branch,
                dataset_hint, hyperparams, requirements, created_by)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (config_id, label, description, script_path, git_repo, git_branch,
             dataset_hint, hyperparams, requirements, created_by),
        )
        return cur.fetchone()


def update(config_id: str, *, label: str, description: str, script_path: str,
           git_repo: str, git_branch: str, dataset_hint: str,
           hyperparams: str, requirements: str, created_by: str) -> dict | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE training_configs
               SET label=%s, description=%s, script_path=%s, git_repo=%s, git_branch=%s,
                   dataset_hint=%s, hyperparams=%s, requirements=%s, created_by=%s, updated_at=now()
               WHERE id=%s RETURNING *""",
            (label, description, script_path, git_repo, git_branch,
             dataset_hint, hyperparams, requirements, created_by, config_id),
        )
        return cur.fetchone()


def delete(config_id: str) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM training_configs WHERE id=%s", (config_id,))
