"""Evaluate WiMose checkpoint on held-out test split (pose + action).

Usage:
    python -m rfpose.evaluation.eval_wimose_action \\
        --checkpoint checkpoints/wimose-mmfi17j-proto1-action-v1/best.pt \\
        --gold-dir data/gold/rfpose-humanlike-v2-proto1/mmfi
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

from rfpose.data.gold_npz_dataset import ACTION_LABELS, GoldNpzDataset, _SubsetGoldNpz
from rfpose.models.wimose_net import WiMoseNet

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_test_set(gold_dir: str | Path, datasets: list[str] | None = None) -> GoldNpzDataset:
    full = GoldNpzDataset(
        gold_dir, split=None, datasets=datasets,
        augment=False, require_pose=True, require_action=False,
    )
    meta_cache: dict[str, np.ndarray | None] = {}

    def _get_meta(ds_name: str) -> np.ndarray | None:
        if ds_name not in meta_cache:
            meta_path = full.gold_dir / ds_name / "metadata.npz"
            meta_cache[ds_name] = (
                np.load(meta_path, allow_pickle=True)["metadata"]
                if meta_path.exists() else None
            )
        return meta_cache[ds_name]

    test_idx: list[int] = []
    for i, entry in enumerate(full.entries):
        meta = _get_meta(entry["dataset"])
        if meta is None:
            continue
        j = entry["index"]
        if j < len(meta) and meta[j].get("split", "") == "test":
            test_idx.append(i)

    if not test_idx:
        raise RuntimeError(f"No test-split samples under {gold_dir}")

    log.info("Test split: %d samples", len(test_idx))
    return _SubsetGoldNpz(full, test_idx, augment=False)


def _prepare_batch(
    batch: dict,
    device: torch.device,
    csi_mean: torch.Tensor | None,
    csi_std: torch.Tensor | None,
    root_joint: int,
    center_pose: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    csi = batch["csi"].to(device, non_blocking=True)
    coords = batch["coords"].to(device, non_blocking=True)
    mask = batch["pose_mask"].to(device, non_blocking=True)
    x = csi.permute(0, 3, 2, 1).contiguous()
    if csi_mean is not None and csi_std is not None:
        x = (x - csi_mean) / csi_std
    t = coords.shape[1]
    gt = coords[:, t // 2, :, :]
    if center_pose and 0 <= root_joint < gt.shape[1]:
        gt = gt - gt[:, root_joint : root_joint + 1, :]
    return x, gt, mask


def load_wimose(checkpoint_path: str, device: torch.device) -> tuple[WiMoseNet, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_joints = int(ckpt.get("n_joints", 17))
    num_actions = int(ckpt.get("num_actions", 0))
    use_fk = bool(ckpt.get("use_fk_head", False))
    use_gcn = bool(ckpt.get("use_gcn_head", False)) and not use_fk

    model = WiMoseNet(
        n_joints=n_joints,
        in_channels=2,
        use_gcn_head=use_gcn,
        use_fk_head=use_fk,
        num_actions=num_actions,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate(
    model: WiMoseNet,
    loader: DataLoader,
    device: torch.device,
    ckpt: dict,
) -> dict:
    csi_mean = ckpt.get("csi_mean")
    csi_std = ckpt.get("csi_std")
    if csi_mean is not None:
        csi_mean = csi_mean.to(device)
    if csi_std is not None:
        csi_std = csi_std.to(device)
    root_joint = int(ckpt.get("root_joint", 0))
    center_pose = bool(ckpt.get("center_pose", True))
    has_action = getattr(model, "action_head", None) is not None

    mpjpe_vals: list[float] = []
    action_preds: list[int] = []
    action_gts: list[int] = []
    per_action_mpjpe: dict[int, list[float]] = defaultdict(list)
    latencies_ms: list[float] = []

    for batch in loader:
        x, gt, mask = _prepare_batch(
            batch, device, csi_mean, csi_std, root_joint, center_pose,
        )
        t0 = time.perf_counter()
        if has_action:
            out = model(x, return_action=True)
            pred = out["coords"]
            action_logits = out["action_logits"]
        else:
            pred = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        valid = mask.bool()
        for i in range(pred.shape[0]):
            if not valid[i]:
                continue
            mpjpe_i = (pred[i] - gt[i]).norm(dim=-1).mean().item()
            mpjpe_vals.append(mpjpe_i)
            if has_action and "action_label" in batch:
                a_ok = float(batch.get("action_mask", torch.ones(1))[i].item())
                if a_ok >= 0.5:
                    act = int(batch["action_label"][i].item())
                    pred_act = int(action_logits[i].argmax().item())
                    action_preds.append(pred_act)
                    action_gts.append(act)
                    per_action_mpjpe[act].append(mpjpe_i)

    report: dict = {
        "n_samples": len(mpjpe_vals),
        "mpjpe_m": float(np.mean(mpjpe_vals)) if mpjpe_vals else float("nan"),
        "mpjpe_mm": float(np.mean(mpjpe_vals) * 1000) if mpjpe_vals else float("nan"),
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "has_action_head": has_action,
    }

    if action_preds:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
        report["action_accuracy"] = float(accuracy_score(action_gts, action_preds))
        prec, rec, f1, _ = precision_recall_fscore_support(
            action_gts, action_preds, average="macro", zero_division=0,
        )
        report["action_macro_f1"] = float(f1)
        report["action_macro_precision"] = float(prec)
        report["action_macro_recall"] = float(rec)
        report["confusion_matrix"] = confusion_matrix(action_gts, action_preds).tolist()

        labels_sorted = sorted(set(action_gts))
        prec_per, rec_per, f1_per, sup = precision_recall_fscore_support(
            action_gts, action_preds, labels=labels_sorted, average=None, zero_division=0,
        )
        per_action: dict = {}
        for label_id, p, r, f, n in zip(labels_sorted, prec_per, rec_per, f1_per, sup):
            name = ACTION_LABELS[label_id] if label_id < len(ACTION_LABELS) else f"class_{label_id}"
            per_action[name] = {
                "f1": float(f),
                "precision": float(p),
                "recall": float(r),
                "support": int(n),
                "mpjpe_mm": float(np.mean(per_action_mpjpe[label_id]) * 1000)
                if per_action_mpjpe[label_id] else None,
            }
        report["per_action"] = per_action

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate WiMose on test split")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--gold-dir",
        default="/mnt/storage_6/project_data/pl0501-01/rf-worldpose/data/gold/rfpose-humanlike-v2-proto1",
    )
    ap.add_argument("--datasets", nargs="*", default=["mmfi"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", default="eval_wimose_action_test.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    log.info("Loading %s", args.checkpoint)
    model, ckpt = load_wimose(args.checkpoint, device)

    gold_path = Path(args.gold_dir)
    if (gold_path / "mmfi").is_dir() and args.datasets == ["mmfi"]:
        gold_path = gold_path  # parent with mmfi subdir
    test_ds = build_test_set(gold_path, datasets=list(args.datasets))
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    report = evaluate(model, loader, device, ckpt)
    report["checkpoint"] = args.checkpoint
    report["gold_dir"] = str(gold_path)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Saved %s", out)
    log.info("MPJPE: %.1f mm | samples: %d", report["mpjpe_mm"], report["n_samples"])
    if "action_accuracy" in report:
        log.info(
            "Action acc: %.3f | macro-F1: %.3f",
            report["action_accuracy"],
            report["action_macro_f1"],
        )


if __name__ == "__main__":
    main()
