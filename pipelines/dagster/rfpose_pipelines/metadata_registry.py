from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text


def postgres_url() -> str:
    if os.getenv("DATABASE_URL"):
        url = os.environ["DATABASE_URL"]
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "rfpose")
    user = os.getenv("POSTGRES_USER", "rfpose")
    password = os.getenv("POSTGRES_PASSWORD", "rfpose")
    return (
        f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{db}"
    )


def make_engine():
    return create_engine(postgres_url(), pool_pre_ping=True)


def _split_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for stats in summary.get("datasets", {}).values():
        for split, count in stats.get("splits", {}).items():
            counts[split] = counts.get(split, 0) + int(count)
    return counts


def _first_x_shape(summary: dict[str, Any]) -> list[int]:
    for stats in summary.get("datasets", {}).values():
        shape = stats.get("x_shape")
        if shape:
            return list(shape)
    return []


def _infer_signal_shape(summary: dict[str, Any]) -> dict[str, int | None]:
    shape = _first_x_shape(summary)
    # Current gold X shape is [samples, channels, window_frames, ..., n_subcarriers].
    # Keep inference defensive because source datasets may evolve.
    return {
        "channels": int(shape[1]) if len(shape) >= 2 else None,
        "n_subcarriers": int(shape[-1]) if len(shape) >= 1 else None,
    }


def build_dataset_metadata(
    *,
    dataset_version: str,
    bronze_uri: str,
    silver_uri: str,
    gold_uri: str,
    silver_report: dict[str, Any],
    gold_summary: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    split_counts = _split_counts(gold_summary)
    signal_shape = _infer_signal_shape(gold_summary)
    quality_status = quality.get("status") or silver_report.get("status") or "unknown"

    return {
        "dataset_version": dataset_version,
        "bronze_uri": bronze_uri,
        "silver_uri": silver_uri,
        "gold_uri": gold_uri,
        "artifact_uri": gold_uri,
        "rows_count": int(silver_report.get("rows", 0)),
        "node_count": int(silver_report.get("node_count", 0)),
        "num_samples": int(gold_summary.get("num_samples", 0)),
        "num_datasets": int(gold_summary.get("num_datasets", 0)),
        "window_frames": int(gold_summary.get("window_frames", 0)),
        "stride": int(gold_summary.get("stride", 0)),
        "n_subcarriers": signal_shape["n_subcarriers"],
        "channels": signal_shape["channels"],
        "splits": split_counts,
        "train_count": split_counts.get("train", 0),
        "val_count": split_counts.get("val", 0),
        "test_count": split_counts.get("test", 0),
        "quality_status": quality_status,
        "silver_quality": silver_report,
        "gold_quality": quality,
        "gold_summary": gold_summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def upsert_dataset_version(
    *,
    dataset_version: str,
    artifact_uri: str,
    quality_report_uri: str | None,
    stats: dict[str, Any],
    preprocess_version: str,
    source_sessions: list[Any] | None = None,
    teacher_version: str | None = None,
    created_by: str | None = None,
) -> None:
    engine = make_engine()
    statement = text(
        """
        INSERT INTO dataset_versions (
            id,
            source_sessions,
            preprocess_version,
            teacher_version,
            artifact_uri,
            stats,
            quality_report_uri,
            created_by
        )
        VALUES (
            :id,
            CAST(:source_sessions AS jsonb),
            :preprocess_version,
            :teacher_version,
            :artifact_uri,
            CAST(:stats AS jsonb),
            :quality_report_uri,
            :created_by
        )
        ON CONFLICT (id) DO UPDATE SET
            source_sessions = EXCLUDED.source_sessions,
            preprocess_version = EXCLUDED.preprocess_version,
            teacher_version = EXCLUDED.teacher_version,
            artifact_uri = EXCLUDED.artifact_uri,
            stats = EXCLUDED.stats,
            quality_report_uri = EXCLUDED.quality_report_uri,
            created_by = EXCLUDED.created_by
        """
    )
    with engine.begin() as conn:
        conn.execute(
            statement,
            {
                "id": dataset_version,
                "source_sessions": json.dumps(source_sessions or []),
                "preprocess_version": preprocess_version,
                "teacher_version": teacher_version,
                "artifact_uri": artifact_uri,
                "stats": json.dumps(stats, sort_keys=True),
                "quality_report_uri": quality_report_uri,
                "created_by": created_by,
            },
        )
