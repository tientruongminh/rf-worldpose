from __future__ import annotations
from dagster import asset, MetadataValue

@asset
def raw_csi_sessions(context):
    """Bronze immutable CSI sessions discovered in object storage."""
    sessions = []
    context.add_output_metadata({"count": len(sessions)})
    return sessions

@asset
def decoded_csi_frames(context, raw_csi_sessions):
    """Silver decoded CSI frames with schema validation."""
    context.add_output_metadata({"rows": 0})
    return []

@asset
def baseline_profiles(context, decoded_csi_frames):
    """Per-room empty-room baseline profiles."""
    return {"profiles": []}

@asset
def synced_multinode_csi(context, decoded_csi_frames):
    """Timestamp-aligned CSI streams across 4 nodes."""
    return []

@asset
def teacher_pose_labels(context):
    """Teacher-generated pose labels from train-time videos."""
    return []

@asset
def aligned_csi_pose(context, synced_multinode_csi, teacher_pose_labels):
    """CSI windows aligned to teacher labels."""
    return []

@asset
def csi_windows(context, aligned_csi_pose, baseline_profiles):
    """Windowed ML samples."""
    return []

@asset
def quality_reports(context, csi_windows):
    """Quality gates for node count, packet drops, FPS, RSSI, timestamps, labels."""
    report = {"status": "stub", "passed": False}
    context.add_output_metadata({"status": MetadataValue.text(report["status"])})
    return report

@asset
def gold_dataset(context, csi_windows, quality_reports):
    """ML-ready train/val/test dataset with manifest/stats/normalization."""
    dataset = {"dataset_version": "stub", "artifact_uri": "s3://rfpose/gold/stub"}
    context.add_output_metadata(dataset)
    return dataset

@asset
def dataset_registry_entry(context, gold_dataset):
    """Dataset registry record creation."""
    return gold_dataset
