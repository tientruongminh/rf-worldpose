from __future__ import annotations

import os
import time
from pathlib import Path

from dagster import MetadataValue, asset

from rfpose_pipelines.etl.bronze_to_silver import bronze_to_silver
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


def _silver_path() -> str:
    return os.getenv(
        "RFPOSE_SILVER_OUT",
        f"s3://{os.getenv('S3_BUCKET', 'rfpose')}/silver/csi_decoded.jsonl",
    )


def _dataset_version() -> str:
    return os.getenv("RFPOSE_DATASET_VERSION", "rfpose-multitask-v1")


def _gold_dir() -> str:
    return os.getenv(
        "RFPOSE_GOLD_DIR",
        f"s3://{os.getenv('S3_BUCKET', 'rfpose')}/gold/{_dataset_version()}",
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
    bronze_root = _bronze_root()
    datasets = parse_dataset_filter(os.getenv("RFPOSE_BRONZE_DATASETS"))
    max_samples = _optional_int_env("RFPOSE_MAX_SAMPLES_PER_DATASET")
    context.log.info(
        "Resolved bronze dataset roots bronze_root=%s datasets=%s max_samples_per_dataset=%s s3_bucket=%s s3_endpoint=%s",
        bronze_root,
        sorted(datasets) if datasets else "all",
        max_samples if max_samples is not None else "all",
        os.getenv("S3_BUCKET", "rfpose"),
        os.getenv("S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL") or "http://207.180.243.242:9000",
    )

    context.add_output_metadata(
        {
            "bronze_root": _uri_metadata(bronze_root),
            "datasets": MetadataValue.json(sorted(datasets) if datasets else "all"),
            "max_samples_per_dataset": max_samples
            if max_samples is not None
            else "all",
        }
    )
    return {
        "bronze_root": bronze_root,
        "datasets": datasets,
        "max_samples_per_dataset": max_samples,
    }


@asset
def silver_csi_rows(context, bronze_dataset_roots):
    silver_out = _silver_path()
    context.log.info(
        "Starting silver_csi_rows bronze_root=%s silver_out=%s datasets=%s max_samples_per_dataset=%s",
        bronze_dataset_roots["bronze_root"],
        silver_out,
        sorted(bronze_dataset_roots["datasets"]) if bronze_dataset_roots["datasets"] else "all",
        bronze_dataset_roots["max_samples_per_dataset"]
        if bronze_dataset_roots["max_samples_per_dataset"] is not None
        else "all",
    )
    report = bronze_to_silver(
        bronze_dataset_roots["bronze_root"],
        silver_out,
        datasets=bronze_dataset_roots["datasets"],
        max_samples_per_dataset=bronze_dataset_roots["max_samples_per_dataset"],
    )

    elapsed = time.time() - t0
    if report.get("skipped"):
        context.log.info("[silver_csi_rows] SKIPPED (output up-to-date, %d rows) in %.1fs",
                         report["rows"], elapsed)
    else:
        context.log.info("[silver_csi_rows] DONE: %d rows, datasets=%s, status=%s in %.1fs",
                         report["rows"], list(report.get("datasets", {}).keys()), report["status"], elapsed)

    context.add_output_metadata(
        {
            "silver_out": _uri_metadata(silver_out),
            "schema_version": report["schema_version"],
            "rows": report["rows"],
            "datasets": MetadataValue.json(report["datasets"]),
            "node_count": report["node_count"],
            "seq_drops_est": report["seq_drops_est"],
            "status": MetadataValue.text(report["status"]),
        }
    )
    context.log.info(
        "Finished silver_csi_rows silver_out=%s rows=%d datasets=%s status=%s",
        silver_out,
        report["rows"],
        report["datasets"],
        report["status"],
    )
    return {"silver_out": silver_out, "quality": report}


@asset
def silver_quality_report(context, silver_csi_rows):
    report = silver_csi_rows["quality"]
    passed = report["status"] == "ok" and report["rows"] > 0
    context.log.info("[silver_quality_report] status=%s, rows=%d, passed=%s",
                     report["status"], report["rows"], passed)
    quality = {**report, "passed": passed}
    context.log.info(
        "Evaluated silver quality passed=%s rows=%d status=%s",
        passed,
        report["rows"],
        report["status"],
    )

    context.add_output_metadata(
        {
            "passed": passed,
            "rows": report["rows"],
            "datasets": MetadataValue.json(report["datasets"]),
            "status": MetadataValue.text("ok" if passed else "failed"),
        }
    )
    return quality


@asset
def gold_multitask_dataset(context, silver_csi_rows, silver_quality_report):
    if not silver_quality_report["passed"]:
        context.log.error("[gold_multitask_dataset] Silver quality check FAILED, aborting")
        raise RuntimeError("Silver quality check failed; refusing to build gold dataset.")

    gold_dir = _gold_dir()
    context.log.info("[gold_multitask_dataset] START silver -> gold: %s -> %s",
                     silver_csi_rows["silver_out"], gold_dir)
    t0 = time.time()
    datasets = parse_dataset_filter(os.getenv("RFPOSE_GOLD_DATASETS"))
    window_frames = int(os.getenv("RFPOSE_WINDOW_FRAMES", "60"))
    stride = int(os.getenv("RFPOSE_STRIDE", "10"))
    max_samples = _optional_int_env("RFPOSE_MAX_SAMPLES_PER_DATASET")
    context.log.info(
        "Starting gold_multitask_dataset silver_path=%s gold_dir=%s dataset_version=%s datasets=%s window_frames=%d stride=%d max_samples_per_dataset=%s",
        silver_csi_rows["silver_out"],
        gold_dir,
        _dataset_version(),
        sorted(datasets) if datasets else "all",
        window_frames,
        stride,
        max_samples if max_samples is not None else "all",
    )

    summary = silver_to_gold(
        silver_csi_rows["silver_out"],
        gold_dir,
        datasets=datasets,
        window_frames=window_frames,
        stride=stride,
        max_samples_per_dataset=max_samples,
    )

    elapsed = time.time() - t0
    if summary.get("skipped"):
        context.log.info("[gold_multitask_dataset] SKIPPED (output up-to-date, %d samples) in %.1fs",
                         summary["num_samples"], elapsed)
    else:
        context.log.info("[gold_multitask_dataset] DONE: %d samples across %d datasets in %.1fs",
                         summary["num_samples"], summary["num_datasets"], elapsed)

    context.add_output_metadata(
        {
            "gold_dir": _uri_metadata(gold_dir),
            "num_samples": summary["num_samples"],
            "num_datasets": summary["num_datasets"],
            "datasets": MetadataValue.json(sorted(summary["datasets"].keys())),
            "window_frames": summary["window_frames"],
            "stride": summary["stride"],
            "summary": _uri_metadata(f"{gold_dir.rstrip('/')}/summary.json"),
            "label_maps": _uri_metadata(f"{gold_dir.rstrip('/')}/label_maps.json"),
        }
    )
    context.log.info(
        "Finished gold_multitask_dataset gold_dir=%s num_samples=%d num_datasets=%d datasets=%s upload=%s",
        gold_dir,
        summary["num_samples"],
        summary["num_datasets"],
        sorted(summary["datasets"].keys()),
        summary.get("upload"),
    )
    return {"gold_dir": gold_dir, "summary": summary}


@asset
def gold_quality_report(context, gold_multitask_dataset):
    context.log.info("[gold_quality_report] Validating gold output at %s",
                     gold_multitask_dataset["gold_dir"])
    gold_dir_value = gold_multitask_dataset["gold_dir"]
    summary = gold_multitask_dataset["summary"]
    is_s3_output = gold_dir_value.startswith("s3://")
    missing = []

    if not is_s3_output:
        gold_dir = Path(gold_dir_value)
        required_root_files = [
            gold_dir / "summary.json",
            gold_dir / "label_maps.json",
        ]
        missing = [str(path) for path in required_root_files if not path.exists()]

    dataset_reports = {}
    for dataset, stats in summary["datasets"].items():
        dataset_missing = []
        if not is_s3_output:
            gold_dir = Path(gold_dir_value)
            dataset_dir = gold_dir if summary["num_datasets"] == 1 else gold_dir / dataset
            required_files = [
                dataset_dir / "x.npz",
                dataset_dir / "y.npz",
                dataset_dir / "metadata.npz",
                dataset_dir / "manifest.json",
                dataset_dir / "stats.json",
                dataset_dir / "normalization.json",
            ]
            dataset_missing = [str(path) for path in required_files if not path.exists()]

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

    upload_ok = True
    if is_s3_output:
        upload_ok = summary.get("upload", {}).get("object_count", 0) > 0

    passed = summary["num_samples"] > 0 and upload_ok and not missing and all(
        report["passed"] for report in dataset_reports.values()
    )
    quality = {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "missing_files": missing,
        "artifact_uri": gold_dir_value,
        "upload": summary.get("upload"),
        "datasets": dataset_reports,
    }
    context.log.info(
        "Evaluated gold quality status=%s passed=%s artifact_uri=%s upload=%s",
        quality["status"],
        passed,
        gold_dir_value,
        summary.get("upload"),
    )

    context.add_output_metadata(
        {
            "status": MetadataValue.text(quality["status"]),
            "passed": passed,
            "missing_files": MetadataValue.json(missing),
            "datasets": MetadataValue.json(dataset_reports),
        }
    )
    return quality


@asset
<<<<<<< HEAD
def dataset_registry_entry(context, gold_multitask_dataset, gold_quality_report):
    context.log.info("[dataset_registry_entry] Registering dataset: quality=%s",
                     gold_quality_report["status"])
=======
def dataset_registry_entry(
    context,
    bronze_dataset_roots,
    silver_csi_rows,
    gold_multitask_dataset,
    gold_quality_report,
):
    dataset_version = _dataset_version()
    artifact_uri = gold_multitask_dataset["gold_dir"]
    quality_report_uri = f"{artifact_uri.rstrip('/')}/summary.json"
    stats = build_dataset_metadata(
        dataset_version=dataset_version,
        bronze_uri=bronze_dataset_roots["bronze_root"],
        silver_uri=silver_csi_rows["silver_out"],
        gold_uri=artifact_uri,
        silver_report=silver_csi_rows["quality"],
        gold_summary=gold_multitask_dataset["summary"],
        quality=gold_quality_report,
    )
>>>>>>> 44c450b (add postgre)
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
        preprocess_version=silver_csi_rows["quality"].get(
            "schema_version", "silver_csi_v1"
        ),
        source_sessions=[],
        teacher_version=os.getenv("RFPOSE_TEACHER_VERSION"),
        created_by=os.getenv("RFPOSE_CREATED_BY", "dagster"),
    )

    context.add_output_metadata(
        {
            "dataset_version": dataset["dataset_version"],
            "artifact_uri": _uri_metadata(dataset["artifact_uri"]),
            "quality_report_uri": _uri_metadata(dataset["quality_report_uri"]),
            "quality_status": MetadataValue.text(gold_quality_report["status"]),
            "num_samples": gold_multitask_dataset["summary"]["num_samples"],
            "rows_count": stats["rows_count"],
            "node_count": stats["node_count"],
            "window_frames": stats["window_frames"],
            "stride": stats["stride"],
            "splits": MetadataValue.json(stats["splits"]),
        }
    )
    context.log.info(
        "Upserted dataset registry entry dataset_version=%s artifact_uri=%s quality_report_uri=%s quality_status=%s rows=%d num_samples=%d",
        dataset["dataset_version"],
        dataset["artifact_uri"],
        dataset["quality_report_uri"],
        gold_quality_report["status"],
        stats["rows_count"],
        gold_multitask_dataset["summary"]["num_samples"],
    )
    return dataset
