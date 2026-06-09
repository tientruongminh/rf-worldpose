"""
eval_v2.py — Evaluation for all model variants (base, rootrel, metafi, gcn_rootrel, subcarrier_attn).

Downloads checkpoint from S3/MLflow, runs inference on val split,
and reports pose metrics + action metrics.

Usage:
    python -m rfpose.evaluation.eval_v2 \
        --s3-path s3://rfpose/mlflow/4/<run_id>/artifacts/best.pt \
        --gold-dir /path/to/gold/data \
        --variant rootrel
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rfpose.data.gold_npz_dataset import (
    GoldNpzDataset, ACTION_LABELS, NUM_ACTIONS,
)
from rfpose.utils.losses import MPJPE, PA_MPJPE

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def download_s3_checkpoint(s3_path: str, local_path: str = "/tmp/best.pt") -> str:
    """Download checkpoint from MinIO/S3."""
    import boto3
    bucket = s3_path.split("/")[2]
    key = "/".join(s3_path.split("/")[3:])
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://207.180.243.242:9000"),
    )
    log.info("Downloading %s → %s", s3_path, local_path)
    s3.download_file(bucket, key, local_path)
    log.info("Downloaded %.1f MB", os.path.getsize(local_path) / 1e6)
    return local_path


def build_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    variant: str | None = None,
):
    """Load tokenizer + model from checkpoint, auto-detecting variant from config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    d = cfg["data"]
    m = cfg["model"]

    detected_variant = m.get("variant", variant or "base")
    if variant and variant != detected_variant:
        log.warning("Overriding detected variant '%s' with '%s'", detected_variant, variant)
        detected_variant = variant

    log.info("Model variant: %s", detected_variant)

    if detected_variant == "rootrel":
        from rfpose.models.csi_tokenizer import CSITokenizer
        from rfpose.models.transformer_rootrel import CSITransformerPoseRootRel

        tokenizer = CSITokenizer(
            n_subcarriers=d["n_subcarriers"],
            patch_size=m["patch_size"],
            d_model=m["d_model"],
            max_seq_len=d.get("window_size", 60) + 10,
            n_nodes=d.get("n_nodes", 1),
            dropout=m.get("dropout", 0.1),
        ).to(device)

        model = CSITransformerPoseRootRel(
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

    elif detected_variant == "metafi":
        from rfpose.models.metafi_baseline import MetaFiTokenizer, MetaFiModel

        tokenizer = MetaFiTokenizer(
            n_subcarriers=d["n_subcarriers"],
            n_channels=2,
            d_model=m["d_model"],
            max_seq_len=d.get("window_size", 60) + 10,
            dropout=m.get("dropout", 0.1),
        ).to(device)

        model = MetaFiModel(
            d_model=m["d_model"],
            n_layers=m.get("n_encoder_layers", 4),
            n_heads=m.get("n_encoder_heads", 8),
            n_joints=d.get("n_joints", 13),
            num_actions=m.get("num_actions", NUM_ACTIONS),
            dropout=m.get("dropout", 0.2),
            ffn_mult=m.get("ffn_mult", 2),
        ).to(device)

    elif detected_variant == "gcn_rootrel":
        from rfpose.models.csi_tokenizer import CSITokenizer
        from rfpose.models.pose_decoder_gcn import CSITransformerPoseGCN

        tokenizer = CSITokenizer(
            n_subcarriers=d["n_subcarriers"],
            patch_size=m["patch_size"],
            d_model=m["d_model"],
            max_seq_len=d.get("window_size", 60) + 10,
            n_nodes=d.get("n_nodes", 1),
            dropout=m.get("dropout", 0.1),
        ).to(device)

        model = CSITransformerPoseGCN(
            n_patches=tokenizer.n_patches,
            d_model=m["d_model"],
            spatial_heads=m["spatial_heads"],
            temporal_heads=m["temporal_heads"],
            n_spatial_layers=m["n_spatial_layers"],
            n_temporal_layers=m["n_temporal_layers"],
            n_gcn_layers=m.get("n_gcn_layers", 3),
            n_gcn_tf_layers=m.get("n_gcn_tf_layers", 3),
            n_joints=d.get("n_joints", 13),
            predict_3d=m.get("predict_3d", True),
            causal_temporal=m.get("causal_temporal", False),
            dropout=m.get("dropout", 0.1),
            ffn_mult=m.get("ffn_mult", 4),
            n_nodes=d.get("n_nodes", 1),
            num_actions=m.get("num_actions", NUM_ACTIONS),
        ).to(device)

    elif detected_variant == "subcarrier_attn":
        from rfpose.models.csi_tokenizer_attn import CSITokenizerAttn
        from rfpose.models.transformer import CSITransformerPose

        n_tokens = m.get("n_tokens", d["n_subcarriers"] // m["patch_size"])
        n_attn_heads = m.get("n_attn_heads", 4)

        tokenizer = CSITokenizerAttn(
            n_subcarriers=d["n_subcarriers"],
            n_tokens=n_tokens,
            d_model=m["d_model"],
            max_seq_len=d.get("window_size", 60) + 10,
            n_nodes=d.get("n_nodes", 1),
            dropout=m.get("dropout", 0.1),
            n_attn_heads=n_attn_heads,
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

    else:
        from rfpose.models.csi_tokenizer import CSITokenizer
        from rfpose.models.transformer import CSITransformerPose

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

    n_params_tok = sum(p.numel() for p in tokenizer.parameters())
    n_params_mod = sum(p.numel() for p in model.parameters())
    log.info("Tokenizer params: %s", f"{n_params_tok:,}")
    log.info("Model params:     %s", f"{n_params_mod:,}")
    log.info("Checkpoint epoch: %s", ckpt.get("epoch", "?"))

    return tokenizer, model, cfg, detected_variant


def build_val_set(gold_dir: str, datasets: list[str] | None = None) -> GoldNpzDataset:
    """Build validation split dataset."""
    full = GoldNpzDataset(
        gold_dir, split=None, datasets=datasets,
        augment=False, require_pose=True, require_action=True,
    )

    meta_cache: dict[str, np.ndarray | None] = {}

    def _get_meta(ds_name: str) -> np.ndarray | None:
        if ds_name not in meta_cache:
            meta_path = full.gold_dir / ds_name / "metadata.npz"
            if meta_path.exists():
                meta_cache[ds_name] = np.load(meta_path, allow_pickle=True)["metadata"]
            else:
                meta_cache[ds_name] = None
        return meta_cache[ds_name]

    val_idx: list[int] = []
    for i, entry in enumerate(full.entries):
        meta = _get_meta(entry["dataset"])
        if meta is not None:
            j = entry["index"]
            if j < len(meta) and meta[j].get("split", "") == "val":
                val_idx.append(i)

    if not val_idx:
        log.warning("No val-split samples; falling back to last 20%%")
        n_val = max(1, len(full) // 5)
        val_idx = list(range(len(full) - n_val, len(full)))

    log.info("Validation split: %d samples", len(val_idx))
    from rfpose.data.gold_npz_dataset import _SubsetGoldNpz
    return _SubsetGoldNpz(full, val_idx, augment=False)


@torch.no_grad()
def evaluate(
    tokenizer: nn.Module,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    variant: str,
    n_joints: int = 13,
) -> dict:
    """Full evaluation: pose + action metrics."""
    mpjpe_fn = MPJPE()
    pa_mpjpe_fn = PA_MPJPE()

    is_rootrel = variant in ("rootrel", "gcn_rootrel")

    all_mpjpe: list[float] = []
    all_pa_mpjpe: list[float] = []
    per_joint_errors: list[np.ndarray] = []
    per_action_mpjpe: dict[int, list[float]] = defaultdict(list)

    action_preds: list[int] = []
    action_gts: list[int] = []

    latencies_ms: list[float] = []
    n_samples = 0

    vis_correct = 0
    vis_total = 0

    for batch in loader:
        csi = batch["csi"].to(device)
        gt_coords = batch["coords"].to(device)
        gt_vis = batch["vis"].to(device)
        B = csi.shape[0]

        t0 = time.perf_counter()
        tokens = tokenizer(csi)
        out = model(tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000
        latencies_ms.append(lat)

        pred_coords = out["coords"]
        if is_rootrel and "root" in out and "offsets" in out:
            pred_coords = out["root"].unsqueeze(2) + out["offsets"]

        mpjpe_val = mpjpe_fn(pred_coords, gt_coords, gt_vis).item()
        pa_mpjpe_val = pa_mpjpe_fn(pred_coords, gt_coords).item()
        all_mpjpe.append(mpjpe_val)
        all_pa_mpjpe.append(pa_mpjpe_val)

        # Per-joint error: [B, T, J]
        joint_err = (pred_coords - gt_coords).norm(dim=-1)  # [B, T, J]
        mean_joint_err = joint_err.mean(dim=(0, 1)).cpu().numpy()  # [J]
        per_joint_errors.append(mean_joint_err)

        # Visibility accuracy
        if "vis_logits" in out:
            vis_pred = (out["vis_logits"] > 0).float()
            vis_correct += (vis_pred == gt_vis).sum().item()
            vis_total += gt_vis.numel()

        # Action classification
        if "action_label" in batch and "action_logits" in out:
            gt_action = batch["action_label"].to(device)
            a_mask = batch.get("action_mask", torch.ones(B, dtype=torch.float32))
            a_mask = a_mask.to(device).bool()
            if a_mask.any():
                preds = out["action_logits"][a_mask].argmax(dim=-1)
                labels = gt_action[a_mask]
                action_preds.extend(preds.cpu().tolist())
                action_gts.extend(labels.cpu().tolist())

                for i in range(B):
                    if a_mask[i]:
                        act = gt_action[i].item()
                        vis_i = gt_vis[i]
                        per_sample = joint_err[i]
                        if vis_i.sum() > 0:
                            m_val = (per_sample * vis_i).sum().item() / vis_i.sum().item()
                        else:
                            m_val = per_sample.mean().item()
                        per_action_mpjpe[act].append(m_val)

        n_samples += B

    report: dict = {
        "n_samples": n_samples,
        "variant": variant,
        "mpjpe_m": float(np.mean(all_mpjpe)),
        "mpjpe_mm": float(np.mean(all_mpjpe)) * 1000,
        "pa_mpjpe_m": float(np.mean(all_pa_mpjpe)),
        "pa_mpjpe_mm": float(np.mean(all_pa_mpjpe)) * 1000,
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "latency_ms_mean": float(np.mean(latencies_ms)),
    }

    # Per-joint breakdown
    if per_joint_errors:
        avg_per_joint = np.mean(per_joint_errors, axis=0) * 1000
        joint_names = [
            "head", "l_shoulder", "l_elbow", "l_wrist",
            "r_shoulder", "r_elbow", "r_wrist",
            "l_hip", "l_knee", "l_ankle",
            "r_hip", "r_knee", "r_ankle",
        ]
        per_joint_report = {}
        for j_idx, j_name in enumerate(joint_names[:len(avg_per_joint)]):
            per_joint_report[j_name] = round(float(avg_per_joint[j_idx]), 1)
        report["per_joint_mpjpe_mm"] = per_joint_report

    # Visibility accuracy
    if vis_total > 0:
        report["vis_accuracy"] = vis_correct / vis_total

    # Action metrics
    if action_preds:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        report["action_accuracy"] = float(accuracy_score(action_gts, action_preds))
        prec, rec, f1, _ = precision_recall_fscore_support(
            action_gts, action_preds, average="macro", zero_division=0,
        )
        report["action_macro_precision"] = float(prec)
        report["action_macro_recall"] = float(rec)
        report["action_macro_f1"] = float(f1)
        report["action_n_classes_seen"] = len(set(action_gts))

        prec_per, rec_per, f1_per, sup = precision_recall_fscore_support(
            action_gts, action_preds, average=None, zero_division=0,
        )
        unique_labels = sorted(set(action_gts + action_preds))
        per_action_report = {}
        for i, label_id in enumerate(unique_labels):
            name = ACTION_LABELS[label_id] if label_id < len(ACTION_LABELS) else f"class_{label_id}"
            per_action_report[name] = {
                "f1": round(float(f1_per[i]), 3),
                "precision": round(float(prec_per[i]), 3),
                "recall": round(float(rec_per[i]), 3),
                "support": int(sup[i]),
                "mpjpe_mm": round(float(np.mean(per_action_mpjpe[label_id])) * 1000, 1)
                    if per_action_mpjpe[label_id] else None,
            }
        report["per_action"] = per_action_report
    else:
        report["action_accuracy"] = None
        report["action_macro_f1"] = None

    return report


def print_report(report: dict):
    """Pretty-print the evaluation report."""
    print()
    print("=" * 60)
    print(f"  EVALUATION REPORT — variant={report['variant']}")
    print("=" * 60)
    print(f"  Samples:     {report['n_samples']}")
    print()
    print("  --- Pose Metrics ---")
    print(f"  MPJPE:       {report['mpjpe_mm']:.1f} mm  ({report['mpjpe_m']:.4f} m)")
    print(f"  PA-MPJPE:    {report['pa_mpjpe_mm']:.1f} mm  ({report['pa_mpjpe_m']:.4f} m)")
    print(f"  Gap:         {report['mpjpe_mm'] - report['pa_mpjpe_mm']:.1f} mm (global translation error)")

    if "vis_accuracy" in report:
        print(f"  Vis Acc:     {report['vis_accuracy']:.4f}")

    print()
    print("  --- Per-Joint MPJPE (mm) ---")
    if "per_joint_mpjpe_mm" in report:
        for jname, jerr in report["per_joint_mpjpe_mm"].items():
            print(f"    {jname:15s} {jerr:7.1f}")

    print()
    print("  --- Action Recognition ---")
    if report.get("action_accuracy") is not None:
        print(f"  Accuracy:    {report['action_accuracy']:.4f}")
        print(f"  Macro F1:    {report['action_macro_f1']:.4f}")
        print(f"  Precision:   {report['action_macro_precision']:.4f}")
        print(f"  Recall:      {report['action_macro_recall']:.4f}")
        print(f"  Classes:     {report['action_n_classes_seen']}")

        if "per_action" in report:
            print()
            print("  --- Per-Action Breakdown ---")
            print(f"  {'Action':20s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'MPJPE':>8s} {'N':>5s}")
            for name, m in sorted(
                report["per_action"].items(),
                key=lambda x: x[1]["support"],
                reverse=True,
            ):
                mpjpe_str = f"{m['mpjpe_mm']:.1f}" if m["mpjpe_mm"] is not None else "n/a"
                print(f"  {name:20s} {m['f1']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {mpjpe_str:>8s} {m['support']:5d}")
    else:
        print("  (No action labels in dataset)")

    print()
    print("  --- Latency ---")
    print(f"  p50:  {report['latency_ms_p50']:.1f} ms/batch")
    print(f"  p95:  {report['latency_ms_p95']:.1f} ms/batch")
    print(f"  mean: {report['latency_ms_mean']:.1f} ms/batch")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Evaluate model variants")
    ap.add_argument("--checkpoint", default=None, help="Local checkpoint path")
    ap.add_argument("--s3-path", default=None, help="S3 path to checkpoint (will download)")
    ap.add_argument("--gold-dir", required=True, help="Gold data directory")
    ap.add_argument("--variant", default=None, help="Model variant override")
    ap.add_argument("--datasets", nargs="*", default=None, help="Specific datasets")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", default="eval_report_v2.json", help="Output JSON")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    ckpt_path = args.checkpoint
    if args.s3_path:
        ckpt_path = download_s3_checkpoint(args.s3_path)
    if not ckpt_path:
        raise ValueError("Provide --checkpoint or --s3-path")

    tokenizer, model, cfg, variant = build_model_from_checkpoint(
        ckpt_path, device, variant=args.variant,
    )

    n_joints = cfg["data"].get("n_joints", 13)
    ds_filter = args.datasets or cfg["data"].get("datasets", None)
    log.info("Gold dir: %s | n_joints: %d | datasets: %s", args.gold_dir, n_joints, ds_filter)

    val_ds = build_val_set(args.gold_dir, datasets=ds_filter)
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    log.info("Running evaluation ...")
    report = evaluate(tokenizer, model, loader, device, variant, n_joints)
    report["checkpoint"] = ckpt_path
    report["gold_dir"] = args.gold_dir
    report["datasets_filter"] = ds_filter

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Report saved: %s", out_path)

    print_report(report)


if __name__ == "__main__":
    main()
