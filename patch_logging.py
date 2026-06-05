#!/usr/bin/env python3
"""Add granular logging to ETL pipeline files."""

import time as _time
import sys

def patch_bronze_to_silver(filepath):
    with open(filepath, "r") as f:
        code = f.read()

    # Ensure time import
    if "import time" not in code:
        code = code.replace("import logging", "import logging\nimport time", 1)
        print("  Added time import")

    # === 1. download_s3_prefix: scan total objects first, then per-50-file progress ===
    old = (
        '    for page in paginator.paginate(**page_kwargs):\n'
        '        for obj in page.get("Contents", []):\n'
        '            key = obj["Key"]\n'
        '            if key.endswith("/"):\n'
        '                continue\n'
        '            if suffixes is not None and Path(key).suffix.lower() not in suffixes:\n'
        '                skipped_count += 1\n'
        '                continue\n'
        '\n'
        '            relative_key = key[len(prefix) :].lstrip("/") if prefix else key\n'
        '            if not relative_key:\n'
        '                continue\n'
        '\n'
        '            local_path = destination / relative_key\n'
        '            local_path.parent.mkdir(parents=True, exist_ok=True)\n'
        '            client.download_file(bucket, key, str(local_path))\n'
        '\n'
        '            object_count += 1\n'
        '            total_bytes += int(obj.get("Size", 0))\n'
        '            latest_key = key\n'
        '            if object_count % 1000 == 0:\n'
        '                log.info(\n'
        '                    "S3 bronze staging progress downloaded=%d skipped=%d bytes=%d latest_key=%s",\n'
        '                    object_count,\n'
        '                    skipped_count,\n'
        '                    total_bytes,\n'
        '                    latest_key,\n'
        '                )'
    )
    new = (
        '    # Pre-scan: count total objects for progress\n'
        '    total_objects = 0\n'
        '    for count_page in paginator.paginate(**page_kwargs):\n'
        '        for obj in count_page.get("Contents", []):\n'
        '            if not obj["Key"].endswith("/"):\n'
        '                total_objects += 1\n'
        '    log.info("  [S3 scan] Found %d total objects in s3://%s/%s", total_objects, bucket, prefix or "")\n'
        '\n'
        '    dl_start = time.time()\n'
        '    for page in paginator.paginate(**page_kwargs):\n'
        '        for obj in page.get("Contents", []):\n'
        '            key = obj["Key"]\n'
        '            if key.endswith("/"):\n'
        '                continue\n'
        '            if suffixes is not None and Path(key).suffix.lower() not in suffixes:\n'
        '                skipped_count += 1\n'
        '                continue\n'
        '\n'
        '            relative_key = key[len(prefix) :].lstrip("/") if prefix else key\n'
        '            if not relative_key:\n'
        '                continue\n'
        '\n'
        '            local_path = destination / relative_key\n'
        '            local_path.parent.mkdir(parents=True, exist_ok=True)\n'
        '            client.download_file(bucket, key, str(local_path))\n'
        '\n'
        '            object_count += 1\n'
        '            total_bytes += int(obj.get("Size", 0))\n'
        '            latest_key = key\n'
        '            if object_count % 50 == 0 or object_count == total_objects:\n'
        '                elapsed = time.time() - dl_start\n'
        '                speed_mb = (total_bytes / 1024 / 1024) / max(elapsed, 0.01)\n'
        '                pct = object_count / max(total_objects, 1) * 100\n'
        '                eta = (elapsed / object_count) * (total_objects - object_count) if object_count > 0 else 0\n'
        '                log.info(\n'
        '                    "  [S3 download] %d/%d (%.0f%%) | %.1f MB | %.2f MB/s | ETA %.0fs | %s",\n'
        '                    object_count, total_objects, pct, total_bytes / 1024 / 1024,\n'
        '                    speed_mb, eta, key.split("/")[-1][:40],\n'
        '                )'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  1. download_s3_prefix: per-50-file progress with speed & ETA")
    else:
        print("  1. download_s3_prefix: SKIP (already patched or mismatch)")

    # === 2. iter_json_packet_rows: count files, per-file progress ===
    old = (
        'def iter_json_packet_rows(bronze_root: str | Path) -> Iterator[dict]:\n'
        '    for file in iter_bronze_batches(bronze_root):\n'
        '        obj = json.loads(file.read_text())\n'
        '        deployment_id = obj.get("deployment_id", "unknown")\n'
        '        for packet in obj.get("packets", []):\n'
        '            yield decode_packet_record(packet, deployment_id, str(file))'
    )
    new = (
        'def iter_json_packet_rows(bronze_root: str | Path) -> Iterator[dict]:\n'
        '    files = list(iter_bronze_batches(bronze_root))\n'
        '    log.info("    [self_captured] Found %d JSON batch files in %s", len(files), bronze_root)\n'
        '    row_count = 0\n'
        '    for fi, file in enumerate(files, 1):\n'
        '        obj = json.loads(file.read_text())\n'
        '        deployment_id = obj.get("deployment_id", "unknown")\n'
        '        packets = obj.get("packets", [])\n'
        '        for packet in packets:\n'
        '            row_count += 1\n'
        '            yield decode_packet_record(packet, deployment_id, str(file))\n'
        '        if fi % 10 == 0 or fi == len(files):\n'
        '            log.info("    [self_captured] File %d/%d (%.0f%%) | %d rows so far | %s",\n'
        '                     fi, len(files), fi / len(files) * 100, row_count, file.name[:30])'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  2. iter_json_packet_rows: per-file progress")
    else:
        print("  2. iter_json_packet_rows: SKIP")

    # === 3. _process_single_dataset: add timing ===
    old = (
        'def _process_single_dataset(args):\n'
        '    """Worker function for parallel dataset processing."""\n'
        '    dataset_name, dataset_root, converter_name, max_samples = args\n'
        '    converters = {\n'
        '        "wimans": iter_wimans_rows,\n'
        '        "person_in_wifi_3d": iter_person_wifi_rows,\n'
        '        "wipose": iter_wipose_rows,\n'
        '        "mmfi": iter_mmfi_rows,\n'
        '        "uthar": iter_uthar_rows,\n'
        '        "wiar": iter_wiar_rows,\n'
        '    }\n'
        '    converter = converters[converter_name]\n'
        '    rows = list(converter(Path(dataset_root), max_samples=max_samples))\n'
        '    return dataset_name, rows'
    )
    new = (
        'def _process_single_dataset(args):\n'
        '    """Worker function for parallel dataset processing."""\n'
        '    dataset_name, dataset_root, converter_name, max_samples = args\n'
        '    converters = {\n'
        '        "wimans": iter_wimans_rows,\n'
        '        "person_in_wifi_3d": iter_person_wifi_rows,\n'
        '        "wipose": iter_wipose_rows,\n'
        '        "mmfi": iter_mmfi_rows,\n'
        '        "uthar": iter_uthar_rows,\n'
        '        "wiar": iter_wiar_rows,\n'
        '    }\n'
        '    converter = converters[converter_name]\n'
        '    t0 = time.time()\n'
        '    log.info("    [%s] WORKER START root=%s max_samples=%s", dataset_name, dataset_root, max_samples)\n'
        '    rows = list(converter(Path(dataset_root), max_samples=max_samples))\n'
        '    elapsed = time.time() - t0\n'
        '    log.info("    [%s] WORKER DONE: %d rows in %.1fs (%.0f rows/s)",\n'
        '             dataset_name, len(rows), elapsed, len(rows) / max(elapsed, 0.01))\n'
        '    return dataset_name, rows'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  3. _process_single_dataset: timing")
    else:
        print("  3. _process_single_dataset: SKIP")

    # === 4. Add per-sample logging to each dataset converter ===
    # iter_person_wifi_rows
    old = (
        'def iter_person_wifi_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    for split in ("train", "test"):\n'
        '        split_dir = root / f"{split}_data"\n'
        '        if not split_dir.exists():\n'
        '            continue\n'
        '\n'
        '        samples = index_person_wifi_samples(\n'
        '            root,\n'
        '            split=split,\n'
        '            single_person_only=False,\n'
        '        )\n'
        '\n'
        '        for sample in limit_samples(samples, max_samples):'
    )
    new = (
        'def iter_person_wifi_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    for split in ("train", "test"):\n'
        '        split_dir = root / f"{split}_data"\n'
        '        if not split_dir.exists():\n'
        '            log.info("    [person_in_wifi_3d] Split %s dir not found, skip", split)\n'
        '            continue\n'
        '\n'
        '        samples = list(index_person_wifi_samples(\n'
        '            root,\n'
        '            split=split,\n'
        '            single_person_only=False,\n'
        '        ))\n'
        '        total = min(len(samples), max_samples) if max_samples else len(samples)\n'
        '        log.info("    [person_in_wifi_3d] Split=%s found %d samples (processing %d)", split, len(samples), total)\n'
        '        t0_pw = time.time()\n'
        '        row_count = 0\n'
        '\n'
        '        for si, sample in enumerate(limit_samples(iter(samples), max_samples), 1):'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  4a. iter_person_wifi_rows: sample counting")
    else:
        print("  4a. iter_person_wifi_rows: SKIP")

    # Add progress to person_wifi after yield block
    old = (
        '                        yield make_silver_row(\n'
        '                            dataset="person_in_wifi_3d",\n'
        '                            sample_id=sample["name"],\n'
        '                            split=split,\n'
        '                            source_file=str(sample["csi_path"]),\n'
        '                            timestamp_us=t,\n'
        '                            seq=t,\n'
        '                            node_id=ant1 * ant2_count + ant2 + 1,\n'
        '                            tx=ant1 + 1,\n'
        '                            rx=ant2 + 1,\n'
        '                            n_subcarriers=n_subcarriers,\n'
        '                            amplitude=to_float_list(amp[ant1, ant2, :, t]),\n'
        '                            phase=to_float_list(phase[ant1, ant2, :, t]),\n'
        '                            num_users=sample["num_people"],\n'
        '                            pose_path=str(sample["keypoint_path"]),\n'
        '                            **pose_fields(common_pose),\n'
        '                        )'
    )
    new = (
        '                        row_count += 1\n'
        '                        yield make_silver_row(\n'
        '                            dataset="person_in_wifi_3d",\n'
        '                            sample_id=sample["name"],\n'
        '                            split=split,\n'
        '                            source_file=str(sample["csi_path"]),\n'
        '                            timestamp_us=t,\n'
        '                            seq=t,\n'
        '                            node_id=ant1 * ant2_count + ant2 + 1,\n'
        '                            tx=ant1 + 1,\n'
        '                            rx=ant2 + 1,\n'
        '                            n_subcarriers=n_subcarriers,\n'
        '                            amplitude=to_float_list(amp[ant1, ant2, :, t]),\n'
        '                            phase=to_float_list(phase[ant1, ant2, :, t]),\n'
        '                            num_users=sample["num_people"],\n'
        '                            pose_path=str(sample["keypoint_path"]),\n'
        '                            **pose_fields(common_pose),\n'
        '                        )\n'
        '            if si % 5 == 0 or si == total:\n'
        '                log.info("    [person_in_wifi_3d/%s] Sample %d/%d (%.0f%%) | %d rows | %.1fs",\n'
        '                         split, si, total, si/total*100, row_count, time.time()-t0_pw)'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  4b. iter_person_wifi_rows: per-sample progress")
    else:
        print("  4b. iter_person_wifi_rows: SKIP")

    # iter_wipose_rows - add sample counting
    old = (
        'def iter_wipose_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    samples = index_wipose_samples(root)\n'
        '\n'
        '    for sample in limit_samples(samples, max_samples):'
    )
    new = (
        'def iter_wipose_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    samples = list(index_wipose_samples(root))\n'
        '    total = min(len(samples), max_samples) if max_samples else len(samples)\n'
        '    log.info("    [wipose] Found %d samples (processing %d)", len(samples), total)\n'
        '    row_count = 0\n'
        '    t0_wp = time.time()\n'
        '\n'
        '    for si, sample in enumerate(limit_samples(iter(samples), max_samples), 1):'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  4c. iter_wipose_rows: sample counting")
    else:
        print("  4c. iter_wipose_rows: SKIP")

    # iter_mmfi_rows - add sample counting
    old = (
        'def iter_mmfi_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    samples = index_mmfi_wifi_csi_samples(root, require_pose=False)\n'
        '\n'
        '    for sample in limit_samples(samples, max_samples):'
    )
    new = (
        'def iter_mmfi_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    samples = list(index_mmfi_wifi_csi_samples(root, require_pose=False))\n'
        '    total = min(len(samples), max_samples) if max_samples else len(samples)\n'
        '    log.info("    [mmfi] Found %d samples (processing %d)", len(samples), total)\n'
        '    row_count = 0\n'
        '    t0_mm = time.time()\n'
        '\n'
        '    for si, sample in enumerate(limit_samples(iter(samples), max_samples), 1):'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  4d. iter_mmfi_rows: sample counting")
    else:
        print("  4d. iter_mmfi_rows: SKIP")

    # iter_uthar_rows - add split progress
    old = (
        'def iter_uthar_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    for split in ("train", "val", "test"):\n'
        '        x, y = load_uthar_arrays(root, split)\n'
        '        sample_count = len(y) if max_samples is None else min(len(y), max_samples)'
    )
    new = (
        'def iter_uthar_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    for split in ("train", "val", "test"):\n'
        '        x, y = load_uthar_arrays(root, split)\n'
        '        sample_count = len(y) if max_samples is None else min(len(y), max_samples)\n'
        '        log.info("    [uthar/%s] %d total samples, processing %d", split, len(y), sample_count)'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  4e. iter_uthar_rows: split progress")
    else:
        print("  4e. iter_uthar_rows: SKIP")

    # iter_wiar_rows - add sample counting
    old = (
        'def iter_wiar_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    samples = index_wiar_samples(root)\n'
        '\n'
        '    for sample in limit_samples(samples, max_samples):'
    )
    new = (
        'def iter_wiar_rows(root: Path, max_samples: int | None = None) -> Iterator[dict]:\n'
        '    samples = list(index_wiar_samples(root))\n'
        '    total = min(len(samples), max_samples) if max_samples else len(samples)\n'
        '    log.info("    [wiar] Found %d samples (processing %d)", len(samples), total)\n'
        '    row_count = 0\n'
        '    t0_wi = time.time()\n'
        '\n'
        '    for si, sample in enumerate(limit_samples(iter(samples), max_samples), 1):'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  4f. iter_wiar_rows: sample counting")
    else:
        print("  4f. iter_wiar_rows: SKIP")

    # === 5. build_quality_report: add step logging ===
    old = (
        'def build_quality_report(rows: list[dict]) -> dict:\n'
        '    node_ids = sorted({row["node_id"] for row in rows if row.get("node_id") is not None})\n'
        '    datasets = Counter(row["dataset"] for row in rows)\n'
        '    seq_drops = 0\n'
        '    seqs_by_stream = defaultdict(list)'
    )
    new = (
        'def build_quality_report(rows: list[dict]) -> dict:\n'
        '    log.info("  [quality_report] Building from %d rows ...", len(rows))\n'
        '    t0_qr = time.time()\n'
        '    node_ids = sorted({row["node_id"] for row in rows if row.get("node_id") is not None})\n'
        '    datasets = Counter(row["dataset"] for row in rows)\n'
        '    log.info("  [quality_report] %d unique nodes, datasets: %s", len(node_ids), dict(datasets))\n'
        '    seq_drops = 0\n'
        '    seqs_by_stream = defaultdict(list)'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  5. build_quality_report: step logging")
    else:
        print("  5. build_quality_report: SKIP")

    # === 6. write_rows: add progress ===
    old = (
        '    out.write_text("\\n".join(json.dumps(row) for row in rows))\n'
        '    log.info("Wrote silver jsonl path=%s rows=%d bytes=%d", out, len(rows), out.stat().st_size)'
    )
    new = (
        '    log.info("  [write_rows] Writing %d rows to %s ...", len(rows), out)\n'
        '    t0_wr = time.time()\n'
        '    out.write_text("\\n".join(json.dumps(row) for row in rows))\n'
        '    elapsed_wr = time.time() - t0_wr\n'
        '    size_mb = out.stat().st_size / 1024 / 1024\n'
        '    log.info("  [write_rows] Done: %d rows, %.1f MB, %.1fs (%.1f MB/s)", len(rows), size_mb, elapsed_wr, size_mb / max(elapsed_wr, 0.01))'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  6. write_rows: progress")
    else:
        print("  6. write_rows: SKIP")

    with open(filepath, "w") as f:
        f.write(code)
    print("  === bronze_to_silver.py SAVED ===")


def patch_silver_to_gold(filepath):
    with open(filepath, "r") as f:
        code = f.read()

    # === 1. load_silver: add detailed logging ===
    old = (
        'def load_silver(path: str | Path) -> list[dict]:\n'
        '    if is_s3_uri(path):\n'
        '        with tempfile.TemporaryDirectory(prefix="rfpose-silver-in-") as tmpdir:\n'
        '            local_path = Path(tmpdir) / Path(str(path)).name\n'
        '            download_s3_file(path, local_path)\n'
        '            return load_silver(local_path)\n'
        '\n'
        '    path = Path(path)\n'
        '\n'
        '    if path.suffix == ".parquet" and pl is not None:\n'
        '        return pl.read_parquet(path).to_dicts()\n'
        '\n'
        '    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]'
    )
    new = (
        'def load_silver(path: str | Path) -> list[dict]:\n'
        '    if is_s3_uri(path):\n'
        '        log.info("  [load_silver] Downloading from S3: %s ...", path)\n'
        '        with tempfile.TemporaryDirectory(prefix="rfpose-silver-in-") as tmpdir:\n'
        '            local_path = Path(tmpdir) / Path(str(path)).name\n'
        '            download_s3_file(path, local_path)\n'
        '            log.info("  [load_silver] Downloaded to %s (%.1f MB)", local_path, local_path.stat().st_size / 1024 / 1024)\n'
        '            return load_silver(local_path)\n'
        '\n'
        '    path = Path(path)\n'
        '    log.info("  [load_silver] Reading local file: %s (%.1f MB)", path, path.stat().st_size / 1024 / 1024 if path.exists() else 0)\n'
        '    t0_ls = time.time()\n'
        '\n'
        '    if path.suffix == ".parquet" and pl is not None:\n'
        '        rows = pl.read_parquet(path).to_dicts()\n'
        '        log.info("  [load_silver] Parsed %d rows from parquet in %.1fs", len(rows), time.time() - t0_ls)\n'
        '        return rows\n'
        '\n'
        '    lines = path.read_text().splitlines()\n'
        '    log.info("  [load_silver] Read %d lines, parsing JSON ...", len(lines))\n'
        '    rows = [json.loads(line) for line in lines if line.strip()]\n'
        '    log.info("  [load_silver] Parsed %d rows in %.1fs", len(rows), time.time() - t0_ls)\n'
        '    return rows'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  1. load_silver: detailed logging")
    else:
        print("  1. load_silver: SKIP")

    # === 2. filter_rows: add before/after counts ===
    old = (
        'def filter_rows(\n'
        '    rows: list[dict],\n'
        '    *,\n'
        '    datasets: set[str] | None,\n'
        '    max_samples_per_dataset: int | None,\n'
        ') -> list[dict]:\n'
        '    if datasets is not None:\n'
        '        rows = [row for row in rows if row.get("dataset") in datasets]\n'
        '\n'
        '    if max_samples_per_dataset is None:\n'
        '        return rows\n'
        '\n'
        '    seen = defaultdict(set)\n'
        '    filtered = []\n'
        '    for row in rows:\n'
        '        dataset = row.get("dataset")\n'
        '        sample_id = str(row.get("sample_id"))\n'
        '        if len(seen[dataset]) >= max_samples_per_dataset and sample_id not in seen[dataset]:\n'
        '            continue\n'
        '        seen[dataset].add(sample_id)\n'
        '        filtered.append(row)\n'
        '    return filtered'
    )
    new = (
        'def filter_rows(\n'
        '    rows: list[dict],\n'
        '    *,\n'
        '    datasets: set[str] | None,\n'
        '    max_samples_per_dataset: int | None,\n'
        ') -> list[dict]:\n'
        '    before = len(rows)\n'
        '    if datasets is not None:\n'
        '        rows = [row for row in rows if row.get("dataset") in datasets]\n'
        '        log.info("  [filter_rows] Dataset filter: %d -> %d rows (kept datasets: %s)", before, len(rows), sorted(datasets))\n'
        '    else:\n'
        '        all_ds = sorted({row.get("dataset") for row in rows})\n'
        '        log.info("  [filter_rows] No dataset filter, all datasets: %s (%d rows)", all_ds, len(rows))\n'
        '\n'
        '    if max_samples_per_dataset is None:\n'
        '        return rows\n'
        '\n'
        '    seen = defaultdict(set)\n'
        '    filtered = []\n'
        '    for row in rows:\n'
        '        dataset = row.get("dataset")\n'
        '        sample_id = str(row.get("sample_id"))\n'
        '        if len(seen[dataset]) >= max_samples_per_dataset and sample_id not in seen[dataset]:\n'
        '            continue\n'
        '        seen[dataset].add(sample_id)\n'
        '        filtered.append(row)\n'
        '    log.info("  [filter_rows] Max samples filter (%d/ds): %d -> %d rows", max_samples_per_dataset, len(rows), len(filtered))\n'
        '    return filtered'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  2. filter_rows: before/after counts")
    else:
        print("  2. filter_rows: SKIP")

    # === 3. group_silver_rows: add group stats ===
    old = (
        'def group_silver_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:\n'
        '    grouped = defaultdict(list)\n'
        '    for row in rows:\n'
        '        if not row.get("dataset") or not row.get("sample_id"):\n'
        '            continue\n'
        '        if not row.get("amplitude"):\n'
        '            continue\n'
        '        grouped[(row["dataset"], str(row["sample_id"]))].append(row)\n'
        '    return dict(grouped)'
    )
    new = (
        'def group_silver_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:\n'
        '    log.info("  [group] Grouping %d rows by (dataset, sample_id) ...", len(rows))\n'
        '    t0_gr = time.time()\n'
        '    grouped = defaultdict(list)\n'
        '    skipped_no_id = 0\n'
        '    skipped_no_amp = 0\n'
        '    for row in rows:\n'
        '        if not row.get("dataset") or not row.get("sample_id"):\n'
        '            skipped_no_id += 1\n'
        '            continue\n'
        '        if not row.get("amplitude"):\n'
        '            skipped_no_amp += 1\n'
        '            continue\n'
        '        grouped[(row["dataset"], str(row["sample_id"]))].append(row)\n'
        '    ds_counts = defaultdict(int)\n'
        '    for (ds, _) in grouped:\n'
        '        ds_counts[ds] += 1\n'
        '    log.info("  [group] %d unique (dataset, sample) pairs in %.1fs | per-dataset: %s | skipped: no_id=%d no_amp=%d",\n'
        '             len(grouped), time.time() - t0_gr, dict(sorted(ds_counts.items())), skipped_no_id, skipped_no_amp)\n'
        '    return dict(grouped)'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  3. group_silver_rows: group stats")
    else:
        print("  3. group_silver_rows: SKIP")

    # === 4. build_gold_records: per-sample progress ===
    old = (
        '    for (dataset, sample_id), rows in sorted(grouped.items()):\n'
        '        rows = sorted(rows, key=lambda row: (int(row["timestamp_us"]), int(row["node_id"])))\n'
        '        first = rows[0]'
    )
    new = (
        '    total_pairs = len(grouped)\n'
        '    log.info("  [build_gold] Processing %d (dataset, sample) pairs into windowed records ...", total_pairs)\n'
        '    t0_bg = time.time()\n'
        '    pair_idx = 0\n'
        '    for (dataset, sample_id), rows in sorted(grouped.items()):\n'
        '        pair_idx += 1\n'
        '        rows = sorted(rows, key=lambda row: (int(row["timestamp_us"]), int(row["node_id"])))\n'
        '        first = rows[0]'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  4a. build_gold_records: progress init")
    else:
        print("  4a. build_gold_records: SKIP")

    # Add per-pair logging after the window append
    old = (
        '            records_by_dataset[dataset].append(\n'
        '                {\n'
        '                    "split": split,\n'
        '                    "sample_id": sample_id,\n'
        '                    "window_start": window_start,\n'
        '                    "x": window_x.astype(np.float32),\n'
        '                    "pose": pose,\n'
        '                    "pose_mask": pose_mask,\n'
        '                    "activity_id": activity_id,\n'
        '                    "activity_mask": int(activity_id >= 0),\n'
        '                    "location_id": location_id,\n'
        '                    "location_mask": int(location_id >= 0),\n'
        '                    "environment_id": environment_id,\n'
        '                    "environment_mask": int(environment_id >= 0),\n'
        '                    "subject_id": subject_id,\n'
        '                    "subject_mask": int(subject_id >= 0),\n'
        '                }\n'
        '            )\n'
        '\n'
        '    return dict(records_by_dataset), label_maps'
    )
    new = (
        '            records_by_dataset[dataset].append(\n'
        '                {\n'
        '                    "split": split,\n'
        '                    "sample_id": sample_id,\n'
        '                    "window_start": window_start,\n'
        '                    "x": window_x.astype(np.float32),\n'
        '                    "pose": pose,\n'
        '                    "pose_mask": pose_mask,\n'
        '                    "activity_id": activity_id,\n'
        '                    "activity_mask": int(activity_id >= 0),\n'
        '                    "location_id": location_id,\n'
        '                    "location_mask": int(location_id >= 0),\n'
        '                    "environment_id": environment_id,\n'
        '                    "environment_mask": int(environment_id >= 0),\n'
        '                    "subject_id": subject_id,\n'
        '                    "subject_mask": int(subject_id >= 0),\n'
        '                }\n'
        '            )\n'
        '        if pair_idx % 50 == 0 or pair_idx == total_pairs:\n'
        '            total_recs = sum(len(v) for v in records_by_dataset.values())\n'
        '            log.info("  [build_gold] Pair %d/%d (%.0f%%) | %d total windows so far | %.1fs",\n'
        '                     pair_idx, total_pairs, pair_idx / total_pairs * 100, total_recs, time.time() - t0_bg)\n'
        '\n'
        '    return dict(records_by_dataset), label_maps'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  4b. build_gold_records: per-pair progress")
    else:
        print("  4b. build_gold_records: SKIP")

    # === 5. write_dataset: add size logging ===
    old = (
        '    all_arrays = records_to_arrays(records, label_maps)\n'
        '    np.savez_compressed(out / "x.npz", X=all_arrays["X"])'
    )
    new = (
        '    log.info("    [write_dataset/%s] Converting %d records to arrays ...", dataset, len(records))\n'
        '    t0_wd = time.time()\n'
        '    all_arrays = records_to_arrays(records, label_maps)\n'
        '    log.info("    [write_dataset/%s] Arrays built: X=%s (%.1f MB) | pose=%s | in %.1fs",\n'
        '             dataset, list(all_arrays["X"].shape),\n'
        '             all_arrays["X"].nbytes / 1024 / 1024,\n'
        '             list(all_arrays["pose"].shape),\n'
        '             time.time() - t0_wd)\n'
        '    log.info("    [write_dataset/%s] Writing x.npz ...", dataset)\n'
        '    np.savez_compressed(out / "x.npz", X=all_arrays["X"])'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  5a. write_dataset: array build logging")
    else:
        print("  5a. write_dataset: SKIP")

    # Add logging after y.npz write
    old = (
        '    np.savez_compressed(\n'
        '        out / "y.npz",\n'
        '        pose=all_arrays["pose"],\n'
        '        pose_mask=all_arrays["pose_mask"],\n'
        '        activity=all_arrays["activity"],\n'
        '        activity_id=all_arrays["activity_id"],\n'
        '        activity_mask=all_arrays["activity_mask"],\n'
        '        location_id=all_arrays["location_id"],\n'
        '        location_mask=all_arrays["location_mask"],\n'
        '        environment_id=all_arrays["environment_id"],\n'
        '        environment_mask=all_arrays["environment_mask"],\n'
        '        subject_id=all_arrays["subject_id"],\n'
        '        subject_mask=all_arrays["subject_mask"],\n'
        '    )\n'
        '    np.savez_compressed(out / "metadata.npz", metadata=metadata)'
    )
    new = (
        '    log.info("    [write_dataset/%s] Writing y.npz ...", dataset)\n'
        '    np.savez_compressed(\n'
        '        out / "y.npz",\n'
        '        pose=all_arrays["pose"],\n'
        '        pose_mask=all_arrays["pose_mask"],\n'
        '        activity=all_arrays["activity"],\n'
        '        activity_id=all_arrays["activity_id"],\n'
        '        activity_mask=all_arrays["activity_mask"],\n'
        '        location_id=all_arrays["location_id"],\n'
        '        location_mask=all_arrays["location_mask"],\n'
        '        environment_id=all_arrays["environment_id"],\n'
        '        environment_mask=all_arrays["environment_mask"],\n'
        '        subject_id=all_arrays["subject_id"],\n'
        '        subject_mask=all_arrays["subject_mask"],\n'
        '    )\n'
        '    log.info("    [write_dataset/%s] Writing metadata.npz ...", dataset)\n'
        '    np.savez_compressed(out / "metadata.npz", metadata=metadata)'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  5b. write_dataset: y.npz and metadata.npz logging")
    else:
        print("  5b. write_dataset: SKIP")

    with open(filepath, "w") as f:
        f.write(code)
    print("  === silver_to_gold.py SAVED ===")


def patch_data_lake(filepath):
    with open(filepath, "r") as f:
        code = f.read()

    # Add detailed step logging to bronze_dataset_roots
    old = (
        '@asset\n'
        'def bronze_dataset_roots(context):\n'
        '    bronze_root = _bronze_root()'
    )
    new = (
        '@asset\n'
        'def bronze_dataset_roots(context):\n'
        '    context.log.info("[STEP 1/6] bronze_dataset_roots: resolving config ...")\n'
        '    bronze_root = _bronze_root()'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  1. bronze_dataset_roots: step label")
    
    # silver_csi_rows
    old = (
        '@asset\n'
        'def silver_csi_rows(context, bronze_dataset_roots):\n'
        '    silver_out = _silver_path()'
    )
    new = (
        '@asset\n'
        'def silver_csi_rows(context, bronze_dataset_roots):\n'
        '    context.log.info("[STEP 2/6] silver_csi_rows: bronze -> silver conversion (HEAVIEST STEP) ...")\n'
        '    silver_out = _silver_path()'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  2. silver_csi_rows: step label")

    # silver_quality_report
    old = (
        '@asset\n'
        'def silver_quality_report(context, silver_csi_rows):'
    )
    new = (
        '@asset\n'
        'def silver_quality_report(context, silver_csi_rows):\n'
        '    context.log.info("[STEP 3/6] silver_quality_report: validating silver output ...")'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  3. silver_quality_report: step label")

    # gold_multitask_dataset
    old = (
        '@asset\n'
        'def gold_multitask_dataset(context, silver_csi_rows, silver_quality_report):'
    )
    new = (
        '@asset\n'
        'def gold_multitask_dataset(context, silver_csi_rows, silver_quality_report):\n'
        '    context.log.info("[STEP 4/6] gold_multitask_dataset: silver -> gold conversion ...")'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  4. gold_multitask_dataset: step label")

    # gold_quality_report
    old = (
        '@asset\n'
        'def gold_quality_report(context, gold_multitask_dataset):'
    )
    new = (
        '@asset\n'
        'def gold_quality_report(context, gold_multitask_dataset):\n'
        '    context.log.info("[STEP 5/6] gold_quality_report: validating gold output ...")'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  5. gold_quality_report: step label")

    # dataset_registry_entry
    old = (
        '@asset\n'
        'def dataset_registry_entry('
    )
    new = (
        '@asset\n'
        'def dataset_registry_entry_asset('
    )
    # Actually don't rename, just add logging
    old = (
        '@asset\ndef dataset_registry_entry(\n'
        '    context,\n'
        '    bronze_dataset_roots,\n'
        '    silver_csi_rows,\n'
        '    gold_multitask_dataset,\n'
        '    gold_quality_report,\n'
        '):'
    )
    new = (
        '@asset\ndef dataset_registry_entry(\n'
        '    context,\n'
        '    bronze_dataset_roots,\n'
        '    silver_csi_rows,\n'
        '    gold_multitask_dataset,\n'
        '    gold_quality_report,\n'
        '):\n'
        '    context.log.info("[STEP 6/6] dataset_registry_entry: registering final dataset ...")'
    )
    if old in code:
        code = code.replace(old, new, 1)
        print("  6. dataset_registry_entry: step label")

    with open(filepath, "w") as f:
        f.write(code)
    print("  === data_lake.py SAVED ===")


if __name__ == "__main__":
    base = "/root/Ngan/rf-worldpose/pipelines/dagster/rfpose_pipelines"
    
    print("=== Patching bronze_to_silver.py ===")
    patch_bronze_to_silver(f"{base}/etl/bronze_to_silver.py")
    
    print("\n=== Patching silver_to_gold.py ===")
    patch_silver_to_gold(f"{base}/etl/silver_to_gold.py")
    
    print("\n=== Patching data_lake.py ===")
    patch_data_lake(f"{base}/assets/data_lake.py")
    
    print("\n=== Syntax check ===")
    import py_compile
    for f in ["etl/bronze_to_silver.py", "etl/silver_to_gold.py", "assets/data_lake.py"]:
        try:
            py_compile.compile(f"{base}/{f}", doraise=True)
            print(f"  {f}: OK")
        except py_compile.PyCompileError as e:
            print(f"  {f}: FAIL - {e}")
