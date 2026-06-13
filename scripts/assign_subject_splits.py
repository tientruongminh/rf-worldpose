#!/usr/bin/env python3
"""Assign train/val/test splits to Gold metadata.

Supports two protocols for MM-Fi:

  Protocol 2 — Cross-subject (default, honest evaluation)
    MM-Fi (sample_id = ``E{e}_S{s}_A{a}``)
    Balanced across all 4 environments:
      test : 2 subjects / env  (subject last digit in {5, 0})  → 8 subjects (~20%)
      val  : 1 subject  / env  (subject last digit == 8)        → 4 subjects (~10%)
      train: remaining 28 subjects                              (~70%)
    Subject never appears in two splits.

  Protocol 1 — Within-subject (sequence-level, matches most papers)
    Split by full sequence id ``E{e}_S{s}_A{a}`` hash → 80/10/10.
    Same subject's different actions can appear in different splits (subject leakage),
    but all windows from the same sequence stay together (no window leakage).
    Gives ~2-3× better MPJPE than cross-subject.

  WiPose (NjtechCVLab)
    Keep the dataset's native Train/Test folders (source_split), then carve a
    deterministic ~10% of the Train *sequences* into val.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np

_MMFI_RE = re.compile(r"E\d+_S(\d+)_A\d+")


def _mmfi_proto1_split(sample_id: str, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> str:
    """Protocol 1: within-subject, sequence-level split.

    Uses MD5 for uniform distribution across the structured MM-Fi id space.
    Every window from the same (env, subject, action) lands in the same split
    (no window leakage), but the same subject's different actions can be in
    different splits (subject leakage — intentional, matching most papers).
    """
    digest = hashlib.md5(f"{seed}:{sample_id}".encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    if frac < test_ratio:
        return "test"
    if frac < test_ratio + val_ratio:
        return "val"
    return "train"


def _mmfi_split(sample_id: str) -> str:
    m = _MMFI_RE.match(sample_id)
    if not m:
        return "train"
    last = int(m.group(1)) % 10  # subject last digit
    if last in (5, 0):
        return "test"
    if last == 8:
        return "val"
    return "train"


def _wipose_split(sample_id: str, source_split: str, val_seq_ids: set[str]) -> str:
    if source_split.lower() == "test":
        return "test"
    return "val" if sample_id in val_seq_ids else "train"


def assign_mmfi(meta: np.ndarray, protocol: int = 2) -> np.ndarray:
    out = []
    for m in meta:
        d = dict(m)
        sid = d.get("sample_id", "")
        if protocol == 1:
            d["split"] = _mmfi_proto1_split(sid)
            d["split_policy"] = "within_subject_sequence_level_proto1"
        else:
            d["split"] = _mmfi_split(sid)
            d["split_policy"] = "cross_subject_balanced_by_environment_proto2"
        out.append(d)
    return np.array(out, dtype=object)


def assign_wipose(meta: np.ndarray, val_ratio: float = 0.1) -> np.ndarray:
    # collect train-folder sequence ids, pick ~val_ratio deterministically by hash
    train_seqs = sorted({
        m["sample_id"] for m in meta
        if str(m.get("source_split", m.get("split", ""))).lower() != "test"
    })
    val_seq_ids = {
        s for s in train_seqs
        if (sum((i + 1) * ord(c) for i, c in enumerate(s)) % 1000) < int(val_ratio * 1000)
    }
    out = []
    for m in meta:
        d = dict(m)
        src = str(d.get("source_split", d.get("split", "")))
        d["split"] = _wipose_split(d.get("sample_id", ""), src, val_seq_ids)
        out.append(d)
    return np.array(out, dtype=object)


def process(ds_dir: Path, dataset: str, protocol: int = 2) -> None:
    meta_path = ds_dir / "metadata.npz"
    backup_name = "metadata.orig.npz" if protocol == 2 else f"metadata.proto{protocol}.orig.npz"
    backup = ds_dir / backup_name
    meta = np.load(meta_path, allow_pickle=True)["metadata"]

    if not backup.exists():
        np.savez_compressed(backup, metadata=meta)
        print(f"[{dataset}] backed up → {backup.name}")

    if dataset == "mmfi":
        new_meta = assign_mmfi(meta, protocol=protocol)
    elif dataset == "wipose":
        new_meta = assign_wipose(meta)
    else:
        raise ValueError(dataset)

    np.savez_compressed(meta_path, metadata=new_meta)
    counts = Counter(m["split"] for m in new_meta)
    print(f"[{dataset}] proto{protocol}  N={len(new_meta)}  splits={dict(counts)}")

    if dataset == "mmfi":
        by_split: dict[str, set[str]] = {}
        for m in new_meta:
            mm = _MMFI_RE.match(m.get("sample_id", ""))
            if mm:
                by_split.setdefault(m["split"], set()).add(mm.group(1))
        for s in ("train", "val", "test"):
            print(f"    {s}: {len(by_split.get(s, set()))} subjects")
        if protocol == 2:
            tr = by_split.get("train", set())
            va = by_split.get("val", set())
            te = by_split.get("test", set())
            assert not (tr & va) and not (tr & te) and not (va & te), "subject leakage!"
            print("    ✓ subject-disjoint")
        else:
            print("    ℹ within-subject: subject leakage is intentional (Protocol 1)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-root", type=Path, required=True,
                    help="e.g. data/gold/rfpose-humanlike-v2")
    ap.add_argument("--datasets", nargs="+", default=["mmfi", "wipose"])
    ap.add_argument("--protocol", type=int, default=2, choices=[1, 2],
                    help="1=within-subject (most papers), 2=cross-subject (honest)")
    args = ap.parse_args()
    for ds in args.datasets:
        ds_dir = args.gold_root / ds
        if (ds_dir / "metadata.npz").exists():
            process(ds_dir, ds, protocol=args.protocol)
        else:
            print(f"[{ds}] skip — no metadata.npz under {ds_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
