"""
eval.py — Standalone evaluation for CSI Transformer Pose model.

Loads a checkpoint, runs inference on the test split of Gold v2 data,
and computes pose + action metrics. Outputs a JSON report compatible
with eval_gate.py, and optionally logs results to MLflow.

Usage:
    python -m rfpose.evaluation.eval --checkpoint checkpoints/best.pt
    python -m rfpose.evaluation.eval --checkpoint best.pt --gold-dir /data/gold/rfpose-unified-v2
    python -m rfpose.evaluation.eval --checkpoint best.pt --mlflow
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
import torch.nn as nn
from torch.utils.data import DataLoader

from rfpose.models.csi_tokenizer import CSITokenizer
from rfpose.models.transformer import CSITransformerPose
from rfpose.data.gold_npz_dataset import (
    GoldNpzDataset, ACTION_LABELS, NUM_ACTIONS,
)
from rfpose.utils.losses import (
    RFPoseLoss, LossConfig, MPJPE, PA_MPJPE, skeleton_for_joints,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def build_test_set(
    gold_dir: str | Path,
    datasets: list[str] | None = None,
) -> GoldNpzDataset:
    """Build a dataset containing only test-split samples."""
    full = GoldNpzDataset(
        gold_dir, split=None, datasets=datasets,
        augment=False, require_pose=False, require_action=False,
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

    test_idx: list[int] = []
    for i, entry in enumerate(full.entries):
        meta = _get_meta(entry["dataset"])
        if meta is not None:
            j = entry["index"]
            if j < len(meta) and meta[j].get("split", "") == "test":
                test_idx.append(i)

    if not test_idx:
        log.warning(
            "No test-split samples found; falling back to last 20%% of data"
        )
        n_test = max(1, len(full) // 5)
        test_idx = list(range(len(full) - n_test, len(full)))

    log.info("Test split: %d samples", len(test_idx))
    from rfpose.data.gold_npz_dataset import _SubsetGoldNpz
    return _SubsetGoldNpz(full, test_idx, augment=False)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[CSITokenizer, CSITransformerPose, dict]:
    """Load tokenizer + model from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    d = cfg["data"]
    m = cfg["model"]

    tokenizer = CSITokenizer(
        n_subcarriers=d["n_subcarriers"],
        patch_size=m["patch_size"],
        d_model=m["d_model"],
        max_seq_len=d.get("window_size", 60) + 10,
        n_nodes=d.get("n_nodes", 1),
        dropout=m.get("dropout", 0.1),
    ).to(device)

    n_joints = d.get("n_joints", 13)

    model = CSITransformerPose(
        n_patches=tokenizer.n_patches,
        d_model=m["d_model"],
        spatial_heads=m["spatial_heads"],
        temporal_heads=m["temporal_heads"],
        n_spatial_layers=m["n_spatial_layers"],
        n_temporal_layers=m["n_temporal_layers"],
        n_decoder_layers=m["n_decoder_layers"],
        n_decoder_temporal_layers=m.get("n_decoder_temporal_layers", 2),
        n_joints=n_joints,
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
def evaluate(
    tokenizer: CSITokenizer,
    model: CSITransformerPose,
    loader: DataLoader,
    device: torch.device,
    n_joints: int = 13,
) -> dict:
    """Run full evaluation and return metrics dict."""
    mpjpe_fn = MPJPE()
    pa_mpjpe_fn = PA_MPJPE()

    bones, sym_pairs = skeleton_for_joints(n_joints)
    loss_cfg = LossConfig(bones=bones, symmetric_pairs=sym_pairs)
    loss_fn = RFPoseLoss(loss_cfg).to(device)
    loss_fn.eval()

    all_mpjpe: list[float] = []
    all_pa_mpjpe: list[float] = []
    all_loss_total: list[float] = []

    per_action_mpjpe: dict[int, list[float]] = defaultdict(list)
    per_action_pa_mpjpe: dict[int, list[float]] = defaultdict(list)
    per_dataset_mpjpe: dict[str, list[float]] = defaultdict(list)

    action_preds: list[int] = []
    action_gts: list[int] = []

    latencies_ms: list[float] = []
    n_samples = 0

    for batch in loader:
        csi = batch["csi"].to(device)
        gt_coords = batch["coords"].to(device)
        gt_vis = batch["vis"].to(device)
        B = csi.shape[0]

        t0 = time.perf_counter()
        tokens = tokenizer(csi)
        out = model(tokens)
        torch.cuda.synchronize() if device.type == "cuda" else None
        lat = (time.perf_counter() - t0) * 1000
        latencies_ms.append(lat)

        pred_dict = {"coords": out["coords"], "vis_logits": out["vis_logits"]}
        gt_dict = {"coords": gt_coords, "vis": gt_vis}
        loss_val, _ = loss_fn(pred_dict, gt_dict)

        mpjpe_val = mpjpe_fn(out["coords"], gt_coords, gt_vis).item()
        pa_mpjpe_val = pa_mpjpe_fn(out["coords"], gt_coords).item()

        all_mpjpe.append(mpjpe_val)
        all_pa_mpjpe.append(pa_mpjpe_val)
        all_loss_total.append(loss_val.item())

        if "action_label" in batch:
            gt_action = batch["action_label"].to(device)
            a_mask = batch.get("action_mask", torch.ones_like(gt_action, dtype=torch.float32))
            a_mask = a_mask.to(device).bool()
            if a_mask.any() and "action_logits" in out:
                preds = out["action_logits"][a_mask].argmax(dim=-1)
                labels = gt_action[a_mask]
                action_preds.extend(preds.cpu().tolist())
                action_gts.extend(labels.cpu().tolist())

            for i in range(B):
                act = gt_action[i].item()
                per_sample_mpjpe = (out["coords"][i] - gt_coords[i]).norm(dim=-1)
                vis_i = gt_vis[i]
                if vis_i.sum() > 0:
                    m_val = (per_sample_mpjpe * vis_i).sum().item() / vis_i.sum().item()
                else:
                    m_val = per_sample_mpjpe.mean().item()
                per_action_mpjpe[act].append(m_val)

        n_samples += B

    report: dict = {
        "n_samples": n_samples,
        "mpjpe": float(np.mean(all_mpjpe)),
        "pa_mpjpe": float(np.mean(all_pa_mpjpe)),
        "loss_total": float(np.mean(all_loss_total)),
        "latency_ms_p50_batch": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95_batch": float(np.percentile(latencies_ms, 95)),
        "latency_ms_mean_batch": float(np.mean(latencies_ms)),
    }

    if action_preds:
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support, confusion_matrix,
        )
        report["accuracy"] = float(accuracy_score(action_gts, action_preds))
        prec, rec, f1, _ = precision_recall_fscore_support(
            action_gts, action_preds, average="macro", zero_division=0,
        )
        report["macro_precision"] = float(prec)
        report["macro_recall"] = float(rec)
        report["macro_f1"] = float(f1)
        report["confusion_matrix"] = confusion_matrix(
            action_gts, action_preds,
        ).tolist()

        prec_per, rec_per, f1_per, sup = precision_recall_fscore_support(
            action_gts, action_preds, average=None, zero_division=0,
        )
        unique_labels = sorted(set(action_gts + action_preds))
        per_action_report = {}
        for i, label_id in enumerate(unique_labels):
            name = ACTION_LABELS[label_id] if label_id < len(ACTION_LABELS) else f"class_{label_id}"
            per_action_report[name] = {
                "precision": float(prec_per[i]),
                "recall": float(rec_per[i]),
                "f1": float(f1_per[i]),
                "support": int(sup[i]),
                "mpjpe": float(np.mean(per_action_mpjpe[label_id]))
                    if per_action_mpjpe[label_id] else None,
            }
        report["per_action"] = per_action_report

    return report


def main():
    ap = argparse.ArgumentParser(description="Evaluate CSI Transformer Pose model")
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    ap.add_argument("--gold-dir", default=None, help="Override gold data directory")
    ap.add_argument("--datasets", nargs="*", default=None, help="Specific datasets to evaluate")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", default="eval_report.json", help="Output JSON path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    args = ap.parse_args()

    device = torch.device(args.device)
    log.info(f"Loading checkpoint: {args.checkpoint}")
    tokenizer, model, cfg = load_model_from_checkpoint(args.checkpoint, device)

    gold_dir = args.gold_dir or cfg["data"]["gold_dir"]
    n_joints = cfg["data"].get("n_joints", 13)
    log.info(f"Gold dir: {gold_dir} | n_joints: {n_joints}")

    log.info("Building test set ...")
    test_ds = build_test_set(gold_dir, datasets=args.datasets)
    log.info(f"Test samples: {len(test_ds)}")

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    log.info("Running evaluation ...")
    report = evaluate(tokenizer, model, loader, device, n_joints)
    report["checkpoint"] = str(args.checkpoint)
    report["gold_dir"] = str(gold_dir)
    report["datasets"] = args.datasets

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info(f"Report saved: {out_path}")

    log.info("=" * 50)
    log.info(f"  MPJPE:       {report['mpjpe']:.4f}")
    log.info(f"  PA-MPJPE:    {report['pa_mpjpe']:.4f}")
    if "accuracy" in report:
        log.info(f"  Action Acc:  {report['accuracy']:.4f}")
        log.info(f"  Macro F1:    {report['macro_f1']:.4f}")
    log.info(f"  Latency p50: {report['latency_ms_p50_batch']:.1f} ms")
    log.info(f"  Latency p95: {report['latency_ms_p95_batch']:.1f} ms")
    log.info(f"  Samples:     {report['n_samples']}")
    log.info("=" * 50)

    if "per_action" in report:
        log.info("Per-action breakdown:")
        for name, m in sorted(report["per_action"].items(), key=lambda x: x[1]["support"], reverse=True):
            mpjpe_str = f"{m['mpjpe']:.4f}" if m["mpjpe"] is not None else "n/a"
            log.info(
                f"  {name:20s}  f1={m['f1']:.3f}  prec={m['precision']:.3f}  "
                f"rec={m['recall']:.3f}  mpjpe={mpjpe_str}  n={m['support']}"
            )

    if args.mlflow:
        import mlflow
        tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "http://207.180.243.242:5000")
        experiment = cfg.get("mlflow", {}).get("experiment_name", "rf-worldpose-transformer")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"eval-{Path(args.checkpoint).stem}"):
            mlflow.log_params({
                "eval_checkpoint": str(args.checkpoint),
                "eval_gold_dir": str(gold_dir),
                "eval_n_samples": report["n_samples"],
            })
            mlflow.log_metrics({
                "test_mpjpe": report["mpjpe"],
                "test_pa_mpjpe": report["pa_mpjpe"],
                "test_loss_total": report["loss_total"],
                "test_latency_p50": report["latency_ms_p50_batch"],
                "test_latency_p95": report["latency_ms_p95_batch"],
            })
            if "accuracy" in report:
                mlflow.log_metrics({
                    "test_accuracy": report["accuracy"],
                    "test_macro_f1": report["macro_f1"],
                    "test_macro_precision": report["macro_precision"],
                    "test_macro_recall": report["macro_recall"],
                })
            mlflow.log_artifact(str(out_path))
            log.info("Results logged to MLflow.")


if __name__ == "__main__":
    main()
