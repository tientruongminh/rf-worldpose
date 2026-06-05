"""Silver → Silver-Unified: flatten spatial dims, pad channels/features, normalize.

Input:  silver_dir with catalog.parquet + csi/<dataset>/<sample>.npy (varying shapes)
Output: unified_dir with unified_catalog.parquet + csi/<sample>.npy (all [2, T, N_padded])
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import logging
import time
import gc

import numpy as np

try:
    import polars as pl
except Exception:
    pl = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

C_UNIFIED = 2  # amplitude + phase (pad if missing)


def load_catalog(silver_dir: Path) -> list[dict]:
    parquet_path = silver_dir / "catalog.parquet"
    jsonl_path = silver_dir / "catalog.jsonl"
    if parquet_path.exists() and pl is not None:
        return pl.read_parquet(parquet_path).to_dicts()
    elif jsonl_path.exists():
        lines = jsonl_path.read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    raise FileNotFoundError(f"No catalog in {silver_dir}")


def compute_n_flat(shape: list[int]) -> int:
    """Given CSI shape [C, T, ...spatial...], compute flat spatial dim N = product of dims[2:]."""
    return int(np.prod(shape[2:]))


def flatten_and_pad(csi: np.ndarray, n_padded: int) -> np.ndarray:
    """Flatten spatial dims and pad channels to C_UNIFIED, features to n_padded.
    
    Input:  [C, T, ...] where C in {1, 2}
    Output: [2, T, n_padded] float32
    """
    C, T = csi.shape[0], csi.shape[1]
    N = int(np.prod(csi.shape[2:]))
    flat = csi.reshape(C, T, N).astype(np.float32)

    out = np.zeros((C_UNIFIED, T, n_padded), dtype=np.float32)
    c_copy = min(C, C_UNIFIED)
    n_copy = min(N, n_padded)
    out[:c_copy, :, :n_copy] = flat[:c_copy, :, :n_copy]
    return out


def normalize_sample(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-score normalize per sample. Returns (normalized, mean, std)."""
    amp = x[0]  # first channel = amplitude
    mean = float(amp.mean())
    std = float(amp.std() + 1e-8)
    x_norm = x.copy()
    x_norm[0] = (amp - mean) / std
    if x.shape[0] > 1:
        phase = x[1]
        p_mean = float(phase.mean())
        p_std = float(phase.std() + 1e-8)
        x_norm[1] = (phase - p_mean) / p_std
    return x_norm, mean, std


def silver_unify(
    silver_dir: str | Path,
    unified_dir: str | Path,
    *,
    min_timesteps: int = 60,
    force: bool = False,
) -> dict:
    silver_dir = Path(silver_dir)
    unified_dir = Path(unified_dir)

    # Idempotent
    report_path = unified_dir / "quality_report.json"
    if not force and report_path.exists():
        try:
            cached = json.loads(report_path.read_text())
            cached["skipped"] = True
            log.info("SKIP silver_unify: output exists at %s", unified_dir)
            return cached
        except Exception:
            pass

    log.info("START silver_unify: %s -> %s (min_timesteps=%d)", silver_dir, unified_dir, min_timesteps)
    t0 = time.time()

    catalog = load_catalog(silver_dir)
    log.info("  Loaded %d samples from catalog", len(catalog))

    # Pass 1: compute N_max from csi_shape
    n_values = []
    for row in catalog:
        shape = json.loads(row["csi_shape"]) if isinstance(row["csi_shape"], str) else row["csi_shape"]
        n_flat = compute_n_flat(shape)
        n_values.append(n_flat)
    n_padded = max(n_values)
    log.info("  N_flat values: min=%d, max=%d, n_padded=%d", min(n_values), max(n_values), n_padded)

    # Pass 2: flatten, pad, normalize, save
    unified_dir.mkdir(parents=True, exist_ok=True)
    csi_out_dir = unified_dir / "csi"
    csi_out_dir.mkdir(exist_ok=True)

    unified_rows = []
    dataset_counts = Counter()
    skipped_short = 0
    skipped_load = 0
    total = len(catalog)

    for idx, row in enumerate(catalog):
        shape = json.loads(row["csi_shape"]) if isinstance(row["csi_shape"], str) else row["csi_shape"]
        T = shape[1]

        if T < min_timesteps:
            skipped_short += 1
            continue

        csi_path = silver_dir / row["csi_path"]
        try:
            csi = np.load(csi_path, mmap_mode="r")
            csi_data = csi[:].astype(np.float32)
            del csi
        except Exception as e:
            skipped_load += 1
            continue

        # Flatten + pad
        unified = flatten_and_pad(csi_data, n_padded)
        del csi_data

        # Normalize
        unified, amp_mean, amp_std = normalize_sample(unified)

        # Save
        ds = row.get("dataset", "unknown")
        sid = row.get("sample_id", f"s{idx}")
        # Use unique filename: dataset_sampleid
        safe_name = f"{ds}_{sid}".replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_path = csi_out_dir / f"{safe_name}.npy"

        if not out_path.exists():
            np.save(out_path, unified)
        del unified

        # Build unified catalog row
        new_row = dict(row)
        new_row["csi_path"] = str(out_path.relative_to(unified_dir))
        new_row["csi_shape"] = json.dumps([C_UNIFIED, T, n_padded])
        new_row["n_flat"] = compute_n_flat(shape)
        new_row["n_padded"] = n_padded
        new_row["amp_mean"] = amp_mean
        new_row["amp_std"] = amp_std
        unified_rows.append(new_row)
        dataset_counts[ds] += 1

        if (idx + 1) % 5000 == 0 or idx + 1 == total:
            log.info("  [%d/%d %.0f%%] %d unified, %d short-skip, %d load-err (%.1fs)",
                     idx + 1, total, (idx + 1) / total * 100,
                     len(unified_rows), skipped_short, skipped_load, time.time() - t0)

    log.info("  Total: %d unified samples, %d short-skipped, %d load-errors",
             len(unified_rows), skipped_short, skipped_load)

    # Write unified catalog
    if pl is not None and unified_rows:
        # Ensure consistent types
        _str_cols = ["dataset", "sample_id", "source_file", "split", "activity",
                     "subject", "environment", "location", "subject_key",
                     "environment_key", "location_key", "csi_path", "csi_shape",
                     "antenna_layout", "pose"]
        _int_cols = ["activity_id", "num_users", "n_tx", "n_rx", "n_antennas",
                     "n_subcarriers", "n_timesteps", "pose_dim", "n_flat", "n_padded"]
        for r in unified_rows:
            for col in _str_cols:
                if col in r and r[col] is None:
                    r[col] = ""
            for col in _int_cols:
                if col in r and r[col] is None:
                    r[col] = -1

        catalog_path = unified_dir / "catalog.parquet"
        df = pl.DataFrame(unified_rows)
        df.write_parquet(catalog_path)
        log.info("  Wrote unified catalog: %s (%d rows, %.1f MB)",
                 catalog_path, len(df), catalog_path.stat().st_size / 1024 / 1024)
    elif unified_rows:
        catalog_path = unified_dir / "catalog.jsonl"
        with open(catalog_path, "w") as f:
            for r in unified_rows:
                f.write(json.dumps(r) + "\n")

    report = {
        "samples": len(unified_rows),
        "datasets": dict(sorted(dataset_counts.items())),
        "n_padded": n_padded,
        "c_unified": C_UNIFIED,
        "min_timesteps": min_timesteps,
        "skipped_short": skipped_short,
        "skipped_load": skipped_load,
        "status": "ok" if unified_rows else "empty",
        "schema_version": "silver_unified_v1",
        "unified_dir": str(unified_dir),
        "skipped": False,
    }
    report_path.write_text(json.dumps(report, indent=2))

    elapsed = time.time() - t0
    log.info("DONE silver_unify: %d samples, n_padded=%d in %.1fs", len(unified_rows), n_padded, elapsed)
    gc.collect()
    return report
