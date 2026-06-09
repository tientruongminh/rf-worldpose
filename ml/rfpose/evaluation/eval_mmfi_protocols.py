"""
eval_mmfi_protocols.py — Evaluate model on MM-Fi dataset using official protocols.

MM-Fi benchmark settings:
    S1 (Random Split):       75/25 random split at sequence level
    S2 (Cross-Subject):      32 train / 8 test subjects (official list)
    S3 (Cross-Environment):  3 rooms train / 1 room test (E04)

Activity protocols:
    P1: 14 daily activities
    P2: 13 rehabilitation exercises
    P3: All 27 actions

Usage:
    python -m rfpose.evaluation.eval_mmfi_protocols \
        --checkpoint checkpoints/best.pt \
        --gold-dir /data/gold/rfpose-unified-v2
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rfpose.models.csi_tokenizer import CSITokenizer
from rfpose.models.transformer import CSITransformerPose
from rfpose.data.gold_npz_dataset import GoldNpzDataset, _SubsetGoldNpz, NUM_ACTIONS
from rfpose.utils.losses import MPJPE, PA_MPJPE

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# ── Official MM-Fi S2 split (cross-subject) ──────────────────────────────
S2_TEST_SUBJECTS = {"S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"}

# ── Official MM-Fi S3 split (cross-environment) ─────────────────────────
S3_TEST_ENVS = {"E04"}

# ── Activity protocols ───────────────────────────────────────────────────
P1_ACTIONS = {  # 14 daily activities
    "A02", "A03", "A04", "A05", "A13", "A14",
    "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A27",
}
P2_ACTIONS = {  # 13 rehabilitation exercises
    "A01", "A06", "A07", "A08", "A09", "A10", "A11", "A12",
    "A15", "A16", "A24", "A25", "A26",
}


def _parse_sample_id(sample_id: str) -> dict:
    """Parse 'E01_S03_A18' into env, subject, action."""
    parts = sample_id.split("_")
    if len(parts) >= 3:
        return {"env": parts[0], "subject": parts[1], "action": parts[2]}
    return {"env": "", "subject": "", "action": ""}


def build_mmfi_split(
    gold_dir: str | Path,
    setting: str,
    protocol: str = "P3",
    split: str = "test",
    seed: int = 42,
) -> tuple[_SubsetGoldNpz, list[dict]]:
    """Build train or test split for MM-Fi evaluation.

    Returns (dataset_subset, metadata_list) where metadata_list has
    per-window {env, subject, action, sample_id}.
    """
    full = GoldNpzDataset(
        gold_dir, split=None, datasets=["mmfi"],
        augment=False, require_pose=True, require_action=False,
    )

    meta_path = Path(gold_dir) / "mmfi" / "metadata.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"MM-Fi metadata not found: {meta_path}")

    raw_meta = np.load(meta_path, allow_pickle=True)["metadata"]

    entry_meta = []
    for entry in full.entries:
        j = entry["index"]
        sid = raw_meta[j].get("sample_id", "") if j < len(raw_meta) else ""
        entry_meta.append({**_parse_sample_id(sid), "sample_id": sid})

    protocol_filter = None
    if protocol == "P1":
        protocol_filter = P1_ACTIONS
    elif protocol == "P2":
        protocol_filter = P2_ACTIONS

    # Group windows by sequence (sample_id) for sequence-level split
    seq_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(entry_meta):
        if protocol_filter and m["action"] not in protocol_filter:
            continue
        seq_to_indices[m["sample_id"]].append(i)

    if setting == "S1":
        # Random 75/25 split at SEQUENCE level
        rng = np.random.RandomState(seed)
        all_seqs = sorted(seq_to_indices.keys())
        rng.shuffle(all_seqs)
        n_test = max(1, len(all_seqs) // 4)
        if split == "test":
            chosen_seqs = set(all_seqs[-n_test:])
        else:
            chosen_seqs = set(all_seqs[:-n_test])

    elif setting == "S2":
        if split == "test":
            chosen_seqs = {s for s in seq_to_indices if entry_meta[seq_to_indices[s][0]]["subject"] in S2_TEST_SUBJECTS}
        else:
            chosen_seqs = {s for s in seq_to_indices if entry_meta[seq_to_indices[s][0]]["subject"] not in S2_TEST_SUBJECTS}

    elif setting == "S3":
        if split == "test":
            chosen_seqs = {s for s in seq_to_indices if entry_meta[seq_to_indices[s][0]]["env"] in S3_TEST_ENVS}
        else:
            chosen_seqs = {s for s in seq_to_indices if entry_meta[seq_to_indices[s][0]]["env"] not in S3_TEST_ENVS}
    else:
        raise ValueError(f"Unknown setting: {setting}")

    indices = []
    meta_out = []
    for seq in sorted(chosen_seqs):
        for idx in seq_to_indices[seq]:
            indices.append(idx)
            meta_out.append(entry_meta[idx])

    log.info(
        "%s/%s split=%s: %d sequences -> %d windows",
        setting, protocol, split, len(chosen_seqs), len(indices),
    )

    ds = _SubsetGoldNpz(full, indices, augment=False)
    return ds, meta_out


def load_model(checkpoint_path: str, device: torch.device):
    """Load tokenizer + model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    d, m = cfg["data"], cfg["model"]

    tokenizer = CSITokenizer(
        n_subcarriers=d["n_subcarriers"],
        patch_size=m["patch_size"],
        d_model=m["d_model"],
        max_seq_len=d.get("window_size", 60) + 10,
        n_nodes=d.get("n_nodes", 1),
        dropout=m.get("dropout", 0.1),
    ).to(device)

    model = CSITransformerPose(
        n_patches=tokenizer.n_patches,
        d_model=m["d_model"],
        spatial_heads=m["spatial_heads"],
        temporal_heads=m["temporal_heads"],
        n_spatial_layers=m["n_spatial_layers"],
        n_temporal_layers=m["n_temporal_layers"],
        n_decoder_layers=m["n_decoder_layers"],
        n_decoder_temporal_layers=m.get("n_decoder_temporal_layers", 2),
        n_joints=d.get("n_joints", 13),
        predict_3d=m.get("predict_3d", True),
        causal_temporal=m.get("causal_temporal", False),
        dropout=m.get("dropout", 0.1),
        ffn_mult=m.get("ffn_mult", 4),
        n_nodes=d.get("n_nodes", 1),
        num_actions=m.get("num_actions", NUM_ACTIONS),
    ).to(device)

    tokenizer.load_state_dict(ckpt["tokenizer"])
    model.load_state_dict(ckpt["model"])
    tokenizer.eval()
    model.eval()
    return tokenizer, model, cfg


@torch.no_grad()
def evaluate_split(
    tokenizer, model, loader, device, n_joints=13,
    per_window_meta: list[dict] | None = None,
) -> dict:
    """Run inference and compute MPJPE / PA-MPJPE.

    If per_window_meta is supplied, also computes per-subject and
    per-action breakdowns.
    """
    mpjpe_fn = MPJPE()
    pa_mpjpe_fn = PA_MPJPE()

    sample_mpjpe: list[float] = []
    sample_pa_mpjpe: list[float] = []

    per_joint_errors: list[np.ndarray] = []  # per-joint MPJPE

    per_subject: dict[str, list[float]] = defaultdict(list)
    per_action: dict[str, list[float]] = defaultdict(list)

    action_preds: list[int] = []
    action_gts: list[int] = []

    window_cursor = 0

    for batch in loader:
        csi = batch["csi"].to(device)
        gt_coords = batch["coords"].to(device)
        gt_vis = batch["vis"].to(device)
        B = csi.shape[0]

        tokens = tokenizer(csi)
        out = model(tokens)

        pred = out["coords"]  # (B, T, J, 3)

        for i in range(B):
            # Per-sample MPJPE (mean over time and joints)
            err = (pred[i] - gt_coords[i]).norm(dim=-1)  # (T, J)
            vis_i = gt_vis[i]  # (T, J)
            if vis_i.sum() > 0:
                m_val = (err * vis_i).sum().item() / vis_i.sum().item()
            else:
                m_val = err.mean().item()
            sample_mpjpe.append(m_val)

            # Per-joint MPJPE (mean over time)
            joint_err = err.mean(dim=0).cpu().numpy()  # (J,)
            per_joint_errors.append(joint_err)

            # PA-MPJPE (mean frame, use middle frame for efficiency)
            mid = pred.shape[1] // 2
            p_frame = pred[i, mid:mid+1].unsqueeze(0)
            g_frame = gt_coords[i, mid:mid+1].unsqueeze(0)
            pa_val = pa_mpjpe_fn(p_frame, g_frame).item()
            sample_pa_mpjpe.append(pa_val)

            if per_window_meta and window_cursor + i < len(per_window_meta):
                meta = per_window_meta[window_cursor + i]
                per_subject[meta["subject"]].append(m_val)
                per_action[meta["action"]].append(m_val)

        # Action accuracy
        if "action_label" in batch and "action_logits" in out:
            gt_a = batch["action_label"].to(device)
            a_mask = batch.get("action_mask", torch.ones_like(gt_a, dtype=torch.float32)).to(device).bool()
            if a_mask.any():
                action_preds.extend(out["action_logits"][a_mask].argmax(dim=-1).cpu().tolist())
                action_gts.extend(gt_a[a_mask].cpu().tolist())

        window_cursor += B

    joint_errors_arr = np.stack(per_joint_errors)  # (N, J)
    mean_per_joint = joint_errors_arr.mean(axis=0) * 1000  # mm

    report = {
        "n_samples": len(sample_mpjpe),
        "mpjpe_mm": float(np.mean(sample_mpjpe) * 1000),
        "pa_mpjpe_mm": float(np.mean(sample_pa_mpjpe) * 1000),
        "std_mpjpe_mm": float(np.std(sample_mpjpe) * 1000),
        "per_joint_mpjpe_mm": mean_per_joint.tolist(),
    }

    if per_subject:
        report["per_subject"] = {
            s: {"mpjpe_mm": float(np.mean(v) * 1000), "n": len(v)}
            for s, v in sorted(per_subject.items())
        }

    if per_action:
        report["per_action"] = {
            a: {"mpjpe_mm": float(np.mean(v) * 1000), "n": len(v)}
            for a, v in sorted(per_action.items())
        }

    if action_preds:
        correct = sum(p == g for p, g in zip(action_preds, action_gts))
        report["action_accuracy"] = correct / len(action_preds)

    return report


def main():
    ap = argparse.ArgumentParser(description="Evaluate on MM-Fi official protocols")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--settings", nargs="*", default=["S1", "S2", "S3"])
    ap.add_argument("--protocols", nargs="*", default=["P3"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", default="eval_mmfi_report.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    log.info("Loading checkpoint: %s", args.checkpoint)
    tokenizer, model, cfg = load_model(args.checkpoint, device)
    n_joints = cfg["data"].get("n_joints", 13)

    results: dict[str, dict] = {"checkpoint": args.checkpoint, "n_joints": n_joints}

    for setting in args.settings:
        for protocol in args.protocols:
            key = f"{setting}_{protocol}"
            log.info("=" * 60)
            log.info("Evaluating: %s / %s", setting, protocol)
            log.info("=" * 60)

            try:
                test_ds, test_meta = build_mmfi_split(
                    args.gold_dir, setting, protocol, split="test",
                )
            except FileNotFoundError as e:
                log.error("Skipping %s: %s", key, e)
                results[key] = {"error": str(e)}
                continue

            if len(test_ds) == 0:
                log.warning("No test samples for %s", key)
                results[key] = {"error": "no test samples"}
                continue

            loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )

            t0 = time.time()
            report = evaluate_split(
                tokenizer, model, loader, device, n_joints, test_meta,
            )
            report["eval_time_s"] = time.time() - t0

            results[key] = report

            log.info(
                "%s: MPJPE=%.1f mm  PA-MPJPE=%.1f mm  (n=%d, %.1fs)",
                key,
                report["mpjpe_mm"],
                report["pa_mpjpe_mm"],
                report["n_samples"],
                report["eval_time_s"],
            )

    # Summary table
    log.info("\n" + "=" * 70)
    log.info("MM-Fi Evaluation Summary (13-joint)")
    log.info("=" * 70)
    log.info("%-12s %12s %12s %8s", "Setting", "MPJPE (mm)", "PA-MPJPE (mm)", "Samples")
    log.info("-" * 50)
    for setting in args.settings:
        for protocol in args.protocols:
            key = f"{setting}_{protocol}"
            r = results.get(key, {})
            if "error" in r:
                log.info("%-12s %s", key, r["error"])
            else:
                log.info(
                    "%-12s %12.1f %12.1f %8d",
                    key, r["mpjpe_mm"], r["pa_mpjpe_mm"], r["n_samples"],
                )

    # Baseline comparison
    baselines_17j = {
        "S1_P3": {"MetaFi++": 197.1, "DT-Pose": 178.5},
        "S2_P3": {"MetaFi++": 231.1, "DT-Pose": 212.8},
        "S3_P3": {"MetaFi++": 369.5, "DT-Pose": 288.6},
        "S1_P1": {"MetaFi++": 186.9, "DT-Pose": 165.3},
        "S2_P1": {"MetaFi++": 222.3, "DT-Pose": 195.6},
        "S3_P1": {"MetaFi++": 367.8, "DT-Pose": 283.0},
    }
    log.info("\nBaseline comparison (17-joint MPJPE from papers):")
    for key, baselines in baselines_17j.items():
        r = results.get(key, {})
        if "mpjpe_mm" in r:
            for name, val in baselines.items():
                delta = r["mpjpe_mm"] - val
                symbol = "▲" if delta > 0 else "▼"
                log.info("  %s vs %s: %.1f vs %.1f (%s%.1f mm)", key, name, r["mpjpe_mm"], val, symbol, abs(delta))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log.info("\nFull report saved: %s", out_path)


if __name__ == "__main__":
    main()
