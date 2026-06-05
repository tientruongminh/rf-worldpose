from __future__ import annotations

import os
import time
from pathlib import Path

from dagster import MetadataValue, asset

from rfpose_pipelines.etl.bronze_to_silver import bronze_to_silver
from rfpose_pipelines.etl.silver_unify import silver_unify
from rfpose_pipelines.etl.silver_to_gold import parse_dataset_filter, silver_to_gold
from rfpose_pipelines.metadata_registry import (
    build_dataset_metadata,
    upsert_dataset_version,
)


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _silver_dir() -> str:
    return os.getenv(
        "RFPOSE_SILVER_OUT",
        "/app/data/silver",
    )


def _dataset_version() -> str:
    return os.getenv("RFPOSE_DATASET_VERSION", "rfpose-multitask-v1")


def _unified_dir() -> str:
    return os.getenv(
        "RFPOSE_UNIFIED_DIR",
        "/app/data/silver_unified",
    )


def _gold_dir() -> str:
    return os.getenv(
        "RFPOSE_GOLD_DIR",
        f"/app/data/gold/{_dataset_version()}",
    )


def _bronze_root() -> str:
    return os.getenv(
        "RFPOSE_BRONZE_ROOT",
        f"s3://{os.getenv('S3_BUCKET', 'rfpose')}/bronze",
    )


def _uri_metadata(value: str):
    if value.startswith("s3://"):
        return MetadataValue.text(value)
    return MetadataValue.path(value)


@asset
def bronze_dataset_roots(context):
    context.log.info("[STEP 1/6] bronze_dataset_roots: resolving config ...")
    bronze_root = _bronze_root()
    datasets = parse_dataset_filter(os.getenv("RFPOSE_BRONZE_DATASETS"))
    max_samples = _optional_int_env("RFPOSE_MAX_SAMPLES_PER_DATASET")
    context.log.info(
        "Resolved bronze_root=%s datasets=%s max_samples=%s",
        bronze_root,
        sorted(datasets) if datasets else "all",
        max_samples if max_samples is not None else "all",
    )
    context.add_output_metadata({
        "bronze_root": _uri_metadata(bronze_root),
        "datasets": MetadataValue.json(sorted(datasets) if datasets else "all"),
        "max_samples_per_dataset": max_samples if max_samples is not None else "all",
    })
    return {
        "bronze_root": bronze_root,
        "datasets": datasets,
        "max_samples_per_dataset": max_samples,
    }


@asset
def silver_csi_rows(context, bronze_dataset_roots):
    context.log.info("[STEP 2/6] silver_csi_rows: bronze -> silver conversion ...")
    silver_out = _silver_dir()
    context.log.info(
        "Starting bronze_to_silver bronze_root=%s silver_out=%s datasets=%s max_samples=%s",
        bronze_dataset_roots["bronze_root"], silver_out,
        sorted(bronze_dataset_roots["datasets"]) if bronze_dataset_roots["datasets"] else "all",
        bronze_dataset_roots["max_samples_per_dataset"] if bronze_dataset_roots["max_samples_per_dataset"] is not None else "all",
    )
    t0 = time.time()
    report = bronze_to_silver(
        bronze_dataset_roots["bronze_root"],
        silver_out,
        datasets=bronze_dataset_roots["datasets"],
        max_samples_per_dataset=bronze_dataset_roots["max_samples_per_dataset"],
    )
    elapsed = time.time() - t0
    context.log.info("[silver_csi_rows] DONE: %d samples, datasets=%s, status=%s in %.1fs",
                     report["samples"], list(report.get("datasets", {}).keys()), report["status"], elapsed)
    context.add_output_metadata({
        "silver_dir": _uri_metadata(silver_out),
        "schema_version": report.get("schema_version", "silver_csi_v2"),
        "samples": report["samples"],
        "datasets": MetadataValue.json(report.get("datasets", {})),
        "status": MetadataValue.text(report["status"]),
    })
    return {"silver_dir": silver_out, "quality": report}


@asset
def silver_quality_report(context, silver_csi_rows):
    context.log.info("[STEP 3/6] silver_quality_report: validating silver output ...")
    report = silver_csi_rows["quality"]
    passed = report["status"] == "ok" and report["samples"] > 0
    context.log.info("[silver_quality_report] status=%s, samples=%d, passed=%s",
                     report["status"], report["samples"], passed)
    quality = {**report, "passed": passed}
    context.add_output_metadata({
        "passed": passed,
        "samples": report["samples"],
        "datasets": MetadataValue.json(report.get("datasets", {})),
        "status": MetadataValue.text("ok" if passed else "failed"),
    })
    return quality


@asset
def silver_unified_dataset(context, silver_csi_rows, silver_quality_report):
    context.log.info("[STEP 4/7] silver_unified_dataset: flatten, pad, normalize ...")
    if not silver_quality_report["passed"]:
        context.log.error("[silver_unified_dataset] Silver quality check FAILED, aborting")
        raise RuntimeError("Silver quality check failed; refusing to build unified dataset.")

    silver_dir = silver_csi_rows["silver_dir"]
    unified_dir = _unified_dir()
    min_timesteps = int(os.getenv("RFPOSE_MIN_TIMESTEPS", "60"))

    context.log.info(
        "Starting silver_unify silver_dir=%s unified_dir=%s min_timesteps=%d",
        silver_dir, unified_dir, min_timesteps,
    )
    t0 = time.time()
    report = silver_unify(silver_dir, unified_dir, min_timesteps=min_timesteps)
    elapsed = time.time() - t0
    context.log.info("[silver_unified_dataset] DONE: %d samples, n_padded=%s in %.1fs",
                     report["samples"], report.get("n_padded"), elapsed)
    context.add_output_metadata({
        "unified_dir": _uri_metadata(unified_dir),
        "samples": report["samples"],
        "n_padded": report.get("n_padded", 0),
        "c_unified": report.get("c_unified", 2),
        "skipped_short": report.get("skipped_short", 0),
        "skipped_load": report.get("skipped_load", 0),
        "datasets": MetadataValue.json(report.get("datasets", {})),
        "status": MetadataValue.text(report.get("status", "unknown")),
    })
    return {"unified_dir": unified_dir, "report": report}


@asset
def gold_multitask_dataset(context, silver_csi_rows, silver_unified_dataset):
    context.log.info("[STEP 5/7] gold_multitask_dataset: unified -> gold conversion ...")

    gold_dir = _gold_dir()
    unified_dir = silver_unified_dataset["unified_dir"]
    silver_dir = silver_csi_rows["silver_dir"]
    datasets = parse_dataset_filter(os.getenv("RFPOSE_GOLD_DATASETS"))
    window_frames = int(os.getenv("RFPOSE_WINDOW_FRAMES", "60"))
    stride = int(os.getenv("RFPOSE_STRIDE", "10"))
    max_samples = _optional_int_env("RFPOSE_MAX_SAMPLES_PER_DATASET")

    context.log.info(
        "Starting silver_to_gold unified_dir=%s silver_dir=%s gold_dir=%s window=%d stride=%d",
        unified_dir, silver_dir, gold_dir, window_frames, stride,
    )
    t0 = time.time()
    summary = silver_to_gold(
        unified_dir, gold_dir,
        silver_dir=silver_dir, datasets=datasets,
        window_frames=window_frames, stride=stride,
        max_samples_per_dataset=max_samples,
    )
    elapsed = time.time() - t0
    context.log.info("[gold_multitask_dataset] DONE: %d windows across %d datasets in %.1fs",
                     summary["num_samples"], summary["num_datasets"], elapsed)
    context.add_output_metadata({
        "gold_dir": _uri_metadata(gold_dir),
        "num_samples": summary["num_samples"],
        "num_datasets": summary["num_datasets"],
        "datasets": MetadataValue.json(sorted(summary["datasets"].keys())),
        "window_frames": summary["window_frames"],
        "stride": summary["stride"],
    })
    return {"gold_dir": gold_dir, "summary": summary}


@asset
def gold_quality_report(context, gold_multitask_dataset):
    context.log.info("[STEP 6/7] gold_quality_report: validating gold output ...")
    gold_dir_value = gold_multitask_dataset["gold_dir"]
    summary = gold_multitask_dataset["summary"]
    missing = []

    if not gold_dir_value.startswith("s3://"):
        gold_dir = Path(gold_dir_value)
        for f in ["summary.json", "label_maps.json"]:
            if not (gold_dir / f).exists():
                missing.append(str(gold_dir / f))

    dataset_reports = {}
    for dataset, stats in summary["datasets"].items():
        dataset_missing = []
        if not gold_dir_value.startswith("s3://"):
            gold_dir = Path(gold_dir_value)
            dataset_dir = gold_dir if summary["num_datasets"] == 1 else gold_dir / dataset
            for f in ["x.npz", "y.npz", "metadata.npz", "manifest.json", "stats.json", "normalization.json"]:
                if not (dataset_dir / f).exists():
                    dataset_missing.append(str(dataset_dir / f))

        pose_shape = stats.get("pose_shape", [])
        pose_ok = len(pose_shape) == 3 and pose_shape[1:] == [13, 3]
        sample_count_ok = stats.get("num_samples", 0) > 0
        dataset_reports[dataset] = {
            "num_samples": stats.get("num_samples", 0),
            "splits": stats.get("splits", {}),
            "x_shape": stats.get("x_shape"),
            "pose_shape": pose_shape,
            "pose_ok": pose_ok,
            "missing_files": dataset_missing,
            "passed": sample_count_ok and pose_ok and not dataset_missing,
        }
        missing.extend(dataset_missing)

    passed = summary["num_samples"] > 0 and not missing and all(r["passed"] for r in dataset_reports.values())
    quality = {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "missing_files": missing,
        "artifact_uri": gold_dir_value,
        "datasets": dataset_reports,
    }
    context.log.info("Gold quality status=%s passed=%s", quality["status"], passed)
    context.add_output_metadata({
        "status": MetadataValue.text(quality["status"]),
        "passed": passed,
        "datasets": MetadataValue.json(dataset_reports),
    })
    return quality


@asset
def dataset_registry_entry(context, bronze_dataset_roots, silver_csi_rows, gold_multitask_dataset, gold_quality_report):
    context.log.info("[STEP 7/7] dataset_registry_entry: registering final dataset ...")
    dataset_version = _dataset_version()
    artifact_uri = gold_multitask_dataset["gold_dir"]
    quality_report_uri = f"{artifact_uri.rstrip('/')}/summary.json"

    stats = build_dataset_metadata(
        dataset_version=dataset_version,
        bronze_uri=bronze_dataset_roots["bronze_root"],
        silver_uri=silver_csi_rows["silver_dir"],
        gold_uri=artifact_uri,
        silver_report=silver_csi_rows["quality"],
        gold_summary=gold_multitask_dataset["summary"],
        quality=gold_quality_report,
    )
    dataset = {
        "dataset_version": dataset_version,
        "artifact_uri": artifact_uri,
        "summary": gold_multitask_dataset["summary"],
        "quality": gold_quality_report,
        "metadata": stats,
        "quality_report_uri": quality_report_uri,
    }
    upsert_dataset_version(
        dataset_version=dataset_version,
        artifact_uri=artifact_uri,
        quality_report_uri=quality_report_uri,
        stats=stats,
        preprocess_version=silver_csi_rows["quality"].get("schema_version", "silver_csi_v2"),
        source_sessions=[],
        teacher_version=os.getenv("RFPOSE_TEACHER_VERSION"),
        created_by=os.getenv("RFPOSE_CREATED_BY", "dagster"),
    )
    context.log.info("Registered dataset_version=%s artifact_uri=%s", dataset_version, artifact_uri)
    context.add_output_metadata({
        "dataset_version": dataset["dataset_version"],
        "artifact_uri": _uri_metadata(dataset["artifact_uri"]),
        "quality_status": MetadataValue.text(gold_quality_report["status"]),
        "num_samples": gold_multitask_dataset["summary"]["num_samples"],
    })
    return dataset
