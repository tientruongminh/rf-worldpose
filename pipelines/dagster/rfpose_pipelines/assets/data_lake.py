from __future__ import annotations
import os
from dagster import asset, MetadataValue
from rfpose_pipelines.etl.bronze_to_silver import bronze_to_silver
from rfpose_pipelines.etl.silver_to_gold import silver_to_gold

@asset
def raw_csi_sessions(context):
    bronze_root = os.getenv('RFPOSE_BRONZE_ROOT', 'data/bronze')
    context.add_output_metadata({'bronze_root': MetadataValue.path(bronze_root)})
    return {'bronze_root': bronze_root}

@asset
def decoded_csi_frames(context, raw_csi_sessions):
    silver_out = os.getenv('RFPOSE_SILVER_OUT', 'data/silver/csi_decoded.parquet')
    report = bronze_to_silver(raw_csi_sessions['bronze_root'], silver_out)
    context.add_output_metadata({'rows': report['rows'], 'node_count': report['node_count'], 'silver_out': MetadataValue.path(silver_out)})
    return {'silver_out': silver_out, 'quality': report}

@asset
def baseline_profiles(context, decoded_csi_frames):
    return {'profiles': [], 'source': decoded_csi_frames['silver_out']}

@asset
def synced_multinode_csi(context, decoded_csi_frames):
    return decoded_csi_frames

@asset
def teacher_pose_labels(context):
    return {'labels': [], 'teacher_version': os.getenv('RFPOSE_TEACHER_VERSION', 'none')}

@asset
def aligned_csi_pose(context, synced_multinode_csi, teacher_pose_labels):
    return {'silver_out': synced_multinode_csi['silver_out'], 'teacher': teacher_pose_labels['teacher_version']}

@asset
def csi_windows(context, aligned_csi_pose, baseline_profiles):
    gold_dir = os.getenv('RFPOSE_GOLD_DIR', 'data/gold/rfpose-local-stub')
    stats = silver_to_gold(aligned_csi_pose['silver_out'], gold_dir)
    context.add_output_metadata({'gold_dir': MetadataValue.path(gold_dir), 'num_samples': stats['num_samples']})
    return {'gold_dir': gold_dir, 'stats': stats}

@asset
def quality_reports(context, csi_windows):
    passed = csi_windows['stats']['num_samples'] > 0
    report = {'status': 'ok' if passed else 'failed', 'passed': passed}
    context.add_output_metadata({'status': MetadataValue.text(report['status'])})
    return report

@asset
def gold_dataset(context, csi_windows, quality_reports):
    dataset = {'dataset_version': os.getenv('RFPOSE_DATASET_VERSION', 'rfpose-local-stub'), 'artifact_uri': csi_windows['gold_dir'], 'stats': csi_windows['stats'], 'quality': quality_reports}
    context.add_output_metadata({'dataset_version': dataset['dataset_version'], 'artifact_uri': MetadataValue.path(dataset['artifact_uri'])})
    return dataset

@asset
def dataset_registry_entry(context, gold_dataset):
    # Control-plane API registration is intentionally optional/offline-safe.
    return gold_dataset
