from dagster import asset

@asset
def raw_csi_sessions():
    """Bronze immutable CSI sessions discovered in object storage."""
    return []

@asset
def decoded_csi_frames(raw_csi_sessions):
    """Silver decoded CSI frames with schema validation."""
    return []

@asset
def gold_dataset(decoded_csi_frames):
    """ML-ready windowed dataset with manifest/stats/normalization."""
    return {"dataset_version": "stub"}
