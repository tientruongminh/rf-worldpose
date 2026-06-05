"""Silver → Silver-Unified: compute unified metadata without copying CSI files.

Scans all Silver .npy shapes, determines N_padded and C_unified,
filters short samples, computes per-sample normalization stats,
and writes a unified_catalog.parquet. No .npy files are created —
the Gold layer applies flatten+pad+normalize on-the-fly during windowing.

Input:  silver_dir with catalog.parquet + csi/<dataset>/<sample>.npy
Output: unified_dir with unified_catalog.parquet + quality_report.json
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import logging
import time

import numpy as np

try:
    import polars as pl
except Exception:
    pl = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

C_UNIFIED = 2


def _npy_shape(path: Path) -> tuple[int, ...] | None:
    """Read .npy shape from header without loading data."""
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, version)
            return shape
    except Exception:
        return None


def load_catalog(silver_dir: Path) -> list[dict]:
    parquet_path = silver_dir / "catalog.parquet"
    jsonl_path = silver_dir / "catalog.jsonl"
    if parquet_path.exists() and pl is not None:
        return pl.read_parquet(parquet_path).to_dicts()
    elif jsonl_path.exists():
        lines = jsonl_path.read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    raise FileNotFoundError(f"No catalog in {silver_dir}")


def compute_n_flat(shape: list[int] | tuple[int, ...]) -> int:
    return int(np.prod(shape[2:]))


def silver_unify(
    silver_dir: str | Path,
    unified_dir: str | Path,
    *,
    min_timesteps: int = 60,
    force: bool = False,
) -> dict:
    silver_dir = Path(silver_dir)
    unified_dir = Path(unified_dir)

    report_path = unified_dir / "quality_report.json"
    if not force and report_path.exists():
        try:
            cached = json.loads(report_path.read_text())
            cached["skipped"] = True
            log.info("SKIP silver_unify: output exists at %s", unified_dir)
            return cached
        except Exception:
            pass

    log.info("START silver_unify: %s -> %s (min_timesteps=%d, catalog-only mode)",
             silver_dir, unified_dir, min_timesteps)
    t0 = time.time()

    catalog = load_catalog(silver_dir)
    log.info("  Loaded %d samples from catalog", len(catalog))

    # Pass 1: compute N_max + filter
    n_values = []
    valid_indices = []
    skipped_short = 0
    skipped_missing = 0

    for idx, row in enumerate(catalog):
        shape_raw = row.get("csi_shape")
        if shape_raw is None:
            skipped_missing += 1
            continue
        shape = json.loads(shape_raw) if isinstance(shape_raw, str) else list(shape_raw)
        T = shape[1] if len(shape) > 1 else 0

        if T < min_timesteps:
            skipped_short += 1
            continue

        csi_path = silver_dir / row["csi_path"]
        if not csi_path.exists():
            skipped_missing += 1
            continue

        n_flat = compute_n_flat(shape)
        n_values.append(n_flat)
        valid_indices.append(idx)

    if not n_values:
        unified_dir.mkdir(parents=True, exist_ok=True)
        report = {"samples": 0, "status": "empty", "skipped": False,
                  "skipped_short": skipped_short, "skipped_missing": skipped_missing}
        report_path.write_text(json.dumps(report, indent=2))
        return report

    n_padded = max(n_values)
    log.info("  N_flat: min=%d, max=%d, n_padded=%d | valid=%d, short=%d, missing=%d (%.1fs)",
             min(n_values), max(n_values), n_padded,
             len(valid_indices), skipped_short, skipped_missing, time.time() - t0)

    # Pass 2: build unified catalog rows (metadata only, no file copies)
    unified_rows = []
    dataset_counts = Counter()
    skipped_load = 0
    total = len(valid_indices)

    for progress_idx, cat_idx in enumerate(valid_indices):
        row = catalog[cat_idx]
        shape = json.loads(row["csi_shape"]) if isinstance(row["csi_shape"], str) else list(row["csi_shape"])
        n_flat = compute_n_flat(shape)
        ds = row.get("dataset", "unknown")

        # Compute normalization stats from a quick header+partial read
        csi_path = silver_dir / row["csi_path"]
        try:
            csi = np.load(csi_path, mmap_mode="r")
            # Sample a small portion for normalization stats
            C = csi.shape[0]
            T = csi.shape[1]
            # Read at most 100 timesteps for stats
            t_sample = min(T, 100)
            chunk = csi[:, :t_sample].astype(np.float32)
            amp_mean = float(chunk[0].mean())
            amp_std = float(chunk[0].std() + 1e-8)
            del chunk, csi
        except Exception:
            skipped_load += 1
            continue

        new_row = dict(row)
        # Keep original csi_path (pointing to silver), add unified metadata
        new_row["original_csi_path"] = row["csi_path"]
        new_row["n_flat"] = n_flat
        new_row["n_padded"] = n_padded
        new_row["c_unified"] = C_UNIFIED
        new_row["amp_mean"] = amp_mean
        new_row["amp_std"] = amp_std
        unified_rows.append(new_row)
        dataset_counts[ds] += 1

        if (progress_idx + 1) % 10000 == 0 or progress_idx + 1 == total:
            log.info("  [%d/%d %.0f%%] %d unified, %d load-err (%.1fs)",
                     progress_idx + 1, total, (progress_idx + 1) / total * 100,
                     len(unified_rows), skipped_load, time.time() - t0)

    log.info("  Total: %d unified, %d short-skipped, %d missing, %d load-errors",
             len(unified_rows), skipped_short, skipped_missing, skipped_load)

    # Write unified catalog
    unified_dir.mkdir(parents=True, exist_ok=True)

    if pl is not None and unified_rows:
        _str_cols = ["dataset", "sample_id", "source_file", "split", "activity",
                     "subject", "environment", "location", "subject_key",
                     "environment_key", "location_key", "csi_path", "csi_shape",
                     "antenna_layout", "pose", "original_csi_path"]
        _int_cols = ["activity_id", "num_users", "n_tx", "n_rx", "n_antennas",
                     "n_subcarriers", "n_timesteps", "pose_dim", "n_flat", "n_padded", "c_unified"]
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
        "skipped_missing": skipped_missing,
        "skipped_load": skipped_load,
        "status": "ok" if unified_rows else "empty",
        "schema_version": "silver_unified_v2_catalog_only",
        "silver_dir": str(silver_dir),
        "unified_dir": str(unified_dir),
        "skipped": False,
    }
    report_path.write_text(json.dumps(report, indent=2))

    elapsed = time.time() - t0
    log.info("DONE silver_unify: %d samples, n_padded=%d in %.1fs (catalog-only, no CSI copies)",
             len(unified_rows), n_padded, elapsed)
    return report
