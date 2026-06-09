"""
eval_job.py — Hydra-based evaluation entry point.

Submittable as a job from the Portal, just like training scripts.
Loads a checkpoint, evaluates on the test split, logs results to MLflow.

Usage (local):
    python -m rfpose.evaluation.eval_job --config-name eval_demo

Usage (via Portal):
    Select config "eval_demo" → Submit → runs on Eagle GPU → results on MLflow
"""
from __future__ import annotations

import logging
import json
import time
from pathlib import Path

import torch
import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from rfpose.evaluation.eval import (
    load_model_from_checkpoint,
    build_test_set,
    evaluate,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="eval_demo", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"\n{'='*60}")
    log.info("RF-WorldPose — Evaluation Job")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"{'='*60}")

    device = torch.device(
        cfg.eval.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )

    checkpoint = cfg.eval.checkpoint
    if not checkpoint:
        log.error("eval.checkpoint is required — path to model .pt file")
        return

    if checkpoint.startswith("mlflow://"):
        run_id = checkpoint.replace("mlflow://", "").split("/")[0]
        artifact = checkpoint.replace(f"mlflow://{run_id}/", "")
        log.info(f"Downloading from MLflow: run={run_id} artifact={artifact}")
        tracking_uri = cfg.mlflow.get("tracking_uri", "http://207.180.243.242:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        local_dir = str(Path("artifacts"))
        Path(local_dir).mkdir(exist_ok=True)
        checkpoint = client.download_artifacts(run_id, artifact, local_dir)
        log.info(f"Downloaded: {checkpoint}")

    log.info(f"Loading checkpoint: {checkpoint}")
    tokenizer, model, model_cfg = load_model_from_checkpoint(checkpoint, device)

    gold_dir = cfg.data.get("gold_dir", model_cfg["data"]["gold_dir"])
    datasets = cfg.data.get("datasets", None)
    if isinstance(datasets, str):
        datasets = [datasets]
    n_joints = model_cfg["data"].get("n_joints", 13)

    log.info(f"Gold dir: {gold_dir}")
    log.info(f"Datasets: {datasets or 'all'}")

    test_ds = build_test_set(gold_dir, datasets=datasets)
    log.info(f"Test samples: {len(test_ds)}")

    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.eval.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.eval.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    log.info("Running evaluation ...")
    t0 = time.time()
    report = evaluate(tokenizer, model, loader, device, n_joints)
    elapsed = time.time() - t0
    report["eval_time_sec"] = elapsed
    report["checkpoint"] = str(checkpoint)
    report["gold_dir"] = str(gold_dir)
    report["datasets"] = datasets

    out_path = Path("eval_report.json")
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
    log.info(f"  Eval time:   {elapsed:.1f}s")
    log.info("=" * 50)

    if "per_action" in report:
        log.info("Per-action breakdown:")
        for name, m in sorted(
            report["per_action"].items(),
            key=lambda x: x[1]["support"],
            reverse=True,
        ):
            mpjpe_str = f"{m['mpjpe']:.4f}" if m["mpjpe"] is not None else "n/a"
            log.info(
                f"  {name:20s}  f1={m['f1']:.3f}  prec={m['precision']:.3f}  "
                f"rec={m['recall']:.3f}  mpjpe={mpjpe_str}  n={m['support']}"
            )

    tracking_uri = cfg.mlflow.get("tracking_uri", "http://207.180.243.242:5000")
    experiment = cfg.mlflow.get("experiment_name", "rf-worldpose-transformer")
    run_name = cfg.mlflow.get("run_name", "eval")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "eval_checkpoint": str(checkpoint),
            "eval_gold_dir": str(gold_dir),
            "eval_datasets": str(datasets),
            "eval_n_samples": report["n_samples"],
        })
        metrics = {
            "test_mpjpe": report["mpjpe"],
            "test_pa_mpjpe": report["pa_mpjpe"],
            "test_loss_total": report["loss_total"],
            "test_latency_p50": report["latency_ms_p50_batch"],
            "test_latency_p95": report["latency_ms_p95_batch"],
            "eval_time_sec": elapsed,
        }
        if "accuracy" in report:
            metrics["test_accuracy"] = report["accuracy"]
            metrics["test_macro_f1"] = report["macro_f1"]
            metrics["test_macro_precision"] = report["macro_precision"]
            metrics["test_macro_recall"] = report["macro_recall"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out_path))
        log.info("Results logged to MLflow.")


if __name__ == "__main__":
    main()
