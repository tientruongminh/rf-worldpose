"""Training script for WiPose (CNN + LSTM + FK).

Follows paper exactly: Section 3 + Section 6.1.
Uses Gold NPZ dataset format with CSI stored as (N, 2, T, 1350).

Usage:
    python -m rfpose.training.train_wipose --config-name wipose_paper_eagle
"""

import os
import json
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.amp as amp
from torch.utils.data import Dataset, DataLoader

import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from rfpose.models.wipose_net import WiPoseNet, SKELETON_CONFIGS
from rfpose.utils.wipose_losses import WiPoseLoss
from rfpose.utils.losses import MPJPE, PA_MPJPE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WiPoseDataset(Dataset):
    """Loads WiPose Gold NPZ data.

    x.npy shape: (N, 2, T, 1350) where channel 0 = amplitude, channel 1 = zeros
    Reshapes to (T, 9, 30, 5) for model input.
    """

    def __init__(
        self,
        gold_dir: str | Path,
        dataset_name: str = "wipose",
        split: str | None = None,
        augment: bool = False,
        csi_shape: tuple = (9, 30, 5),
    ):
        self.gold_dir = Path(gold_dir)
        self.augment = augment
        self.csi_shape = csi_shape
        ds_dir = self.gold_dir / dataset_name

        self.x_path = str(ds_dir / "x.npy")
        self.y_path = str(ds_dir / "y.npz")

        x_mmap = np.load(self.x_path, mmap_mode="r")
        self.n_total = x_mmap.shape[0]
        del x_mmap

        meta_path = ds_dir / "metadata.npz"
        self.indices = list(range(self.n_total))
        if split and meta_path.exists():
            meta = np.load(meta_path, allow_pickle=True)["metadata"]
            self.indices = [
                i for i in range(min(self.n_total, len(meta)))
                if meta[i].get("split", "") == split
            ]

        ref_path = ds_dir / "ref_skeleton.npy"
        if ref_path.exists():
            self.ref_skeleton = np.load(ref_path).astype(np.float32)
        else:
            self.ref_skeleton = None

        log.info("WiPoseDataset: %d windows (split=%s)", len(self.indices), split or "all")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        x_mmap = np.load(self.x_path, mmap_mode="r")
        x_win = np.array(x_mmap[real_idx], dtype=np.float32)
        del x_mmap

        np.nan_to_num(x_win, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        amp_flat = x_win[0]  # (T, N_flat)
        csi = amp_flat.reshape(amp_flat.shape[0], *self.csi_shape)  # (T, *csi_shape)

        y = np.load(self.y_path)
        coords = np.array(y["pose"][real_idx], dtype=np.float32)  # (T, 18, 3)
        y.close()

        csi_t = torch.from_numpy(csi)
        coords_t = torch.from_numpy(coords)

        if self.augment:
            if torch.rand(1) < 0.5:
                csi_t = csi_t + torch.randn_like(csi_t) * 0.03

        return {
            "csi": csi_t,       # (T, 9, 30, 5)
            "coords": coords_t, # (T, 18, 3)
        }


# ---------------------------------------------------------------------------
# Compute reference bone offsets from training data
# ---------------------------------------------------------------------------
def compute_ref_offsets(gold_dir: Path, dataset_name: str = "wipose", n_joints: int = 18) -> torch.Tensor:
    """Compute mean bone offsets from training data for FK layer."""
    ds_dir = gold_dir / dataset_name
    ref_path = ds_dir / "ref_skeleton.npy"

    if ref_path.exists():
        ref = np.load(ref_path)
        log.info("Loaded reference skeleton from %s", ref_path)
    else:
        y = np.load(ds_dir / "y.npz")
        poses = y["pose"]
        mid_t = poses.shape[1] // 2
        mid_poses = poses[:, mid_t]
        ref = mid_poses.mean(axis=0).astype(np.float32)
        y.close()
        np.save(ref_path, ref)
        log.info("Computed reference skeleton: %s", ref.shape)

    _, parent_map, _ = SKELETON_CONFIGS[n_joints]
    offsets = np.zeros((n_joints, 3), dtype=np.float32)
    for child, parent in parent_map.items():
        offsets[child] = ref[child] - ref[parent]
    return torch.from_numpy(offsets)


# ---------------------------------------------------------------------------
# Training / Eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: WiPoseNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: WiPoseLoss,
    ref_offsets: torch.Tensor,
    device: torch.device,
    scaler: amp.GradScaler,
    cfg: DictConfig,
    epoch: int,
) -> dict[str, float]:

    model.train()

    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        csi = batch["csi"].to(device, non_blocking=True)
        gt = batch["coords"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=cfg.training.amp):
            out = model(csi, ref_offsets)
            loss, breakdown = loss_fn(out["coords"], gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k, v in breakdown.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

        if batch_idx % cfg.training.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            log.info(
                "Epoch %d [%d/%d] loss=%.4f pos=%.4f smooth=%.4f rot=%.4f lr=%.6f",
                epoch, batch_idx, len(loader),
                breakdown["loss_total"],
                breakdown["loss_position"],
                breakdown["loss_smooth"],
                breakdown["loss_rotation"],
                lr,
            )
            mlflow.log_metrics(
                {"step_loss": breakdown["loss_total"], "lr": lr},
                step=epoch * len(loader) + batch_idx,
            )

        if cfg.training.dry_run:
            break

    return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}


@torch.no_grad()
def eval_one_epoch(
    model: WiPoseNet,
    loader: DataLoader,
    loss_fn: WiPoseLoss,
    ref_offsets: torch.Tensor,
    device: torch.device,
    cfg: DictConfig,
) -> dict[str, float]:

    model.eval()
    mpjpe_fn = MPJPE()
    pa_mpjpe_fn = PA_MPJPE()

    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        csi = batch["csi"].to(device)
        gt = batch["coords"].to(device)

        out = model(csi, ref_offsets)
        _, breakdown = loss_fn(out["coords"], gt)

        vis = torch.ones(gt.shape[:-1], device=device)
        mpjpe = mpjpe_fn(out["coords"], gt, vis)
        pa_mpjpe = pa_mpjpe_fn(out["coords"], gt)

        breakdown["mpjpe"] = mpjpe.item()
        breakdown["pa_mpjpe"] = pa_mpjpe.item()

        for k, v in breakdown.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

        if cfg.training.dry_run:
            break

    return {f"val_{k}": v / max(n_batches, 1) for k, v in total_metrics.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra.main(config_path="../../configs", config_name="wipose_paper_eagle", version_base=None)
def train(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("WiPose Training (CNN + LSTM + FK)")
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("=" * 60)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed)

    # Data
    gold_dir = Path(cfg.data.gold_dir)
    ds_name = cfg.data.datasets[0] if cfg.data.datasets else "wipose"
    csi_shape = (cfg.model.n_antennas, cfg.model.n_sub, cfg.model.n_packets)

    train_ds = WiPoseDataset(gold_dir, ds_name, split="train", augment=cfg.data.augment, csi_shape=csi_shape)
    val_ds = WiPoseDataset(gold_dir, ds_name, split="val", augment=False, csi_shape=csi_shape)

    if len(val_ds) == 0:
        val_ds = WiPoseDataset(gold_dir, ds_name, split="test", augment=False, csi_shape=csi_shape)

    pin = device.type == "cuda"
    nw = cfg.training.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True,
        persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size * 2, shuffle=False,
        num_workers=nw, pin_memory=pin,
        persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None,
    )
    log.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    # Reference skeleton offsets
    ref_offsets = compute_ref_offsets(gold_dir, ds_name, n_joints=cfg.data.n_joints).to(device)
    log.info("Reference offsets computed: %s", ref_offsets.shape)

    # Model
    model = WiPoseNet(
        n_joints=cfg.data.n_joints,
        n_antennas=cfg.model.n_antennas,
        n_sub=cfg.model.n_sub,
        n_packets=cfg.model.n_packets,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        lstm_dropout=cfg.model.lstm_dropout,
        cnn_dropout=cfg.model.cnn_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model params: %d", n_params)

    # Loss (paper: β=γ=1.0)
    loss_fn = WiPoseLoss(
        beta=cfg.loss.beta,
        gamma=cfg.loss.gamma,
        n_joints=cfg.data.n_joints,
    ).to(device)

    # Optimizer (paper: Adam)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler
    total_steps = cfg.training.epochs * len(train_loader)
    warmup_steps = cfg.training.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = amp.GradScaler(device=cfg.training.device, enabled=cfg.training.amp)

    # MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    best_mpjpe = float("inf")

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))
        mlflow.log_params(OmegaConf.to_container(cfg.training, resolve=True))
        mlflow.log_params(OmegaConf.to_container(cfg.loss, resolve=True))
        mlflow.log_param("architecture", "wipose_cnn_lstm_fk")
        mlflow.log_param("n_params", n_params)

        patience_counter = 0

        for epoch in range(cfg.training.epochs):
            t0 = time.time()

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                loss_fn, ref_offsets, device, scaler, cfg, epoch,
            )

            val_metrics = eval_one_epoch(
                model, val_loader, loss_fn, ref_offsets, device, cfg,
            )

            epoch_time = time.time() - t0
            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "epoch_time": epoch_time}
            mlflow.log_metrics(all_metrics, step=epoch)

            log.info(
                "Epoch %03d/%d [%.1fs] train_loss=%.4f "
                "val_mpjpe=%.4f val_pa_mpjpe=%.4f val_loss=%.4f",
                epoch, cfg.training.epochs, epoch_time,
                train_metrics["loss_total"],
                val_metrics["val_mpjpe"],
                val_metrics["val_pa_mpjpe"],
                val_metrics["val_loss_total"],
            )

            val_mpjpe = val_metrics["val_mpjpe"]
            if val_mpjpe < best_mpjpe:
                best_mpjpe = val_mpjpe
                patience_counter = 0
                ckpt_path = "checkpoints/wipose_best.pt"
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ref_offsets": ref_offsets.cpu(),
                    "metrics": all_metrics,
                    "config": OmegaConf.to_container(cfg),
                }, ckpt_path)
                try:
                    mlflow.log_artifact(ckpt_path)
                except Exception:
                    log.warning("  MLflow artifact upload failed (no S3 creds), checkpoint saved locally")
                log.info("  New best MPJPE: %.4f", best_mpjpe)
            else:
                patience_counter += 1
                log.info("  No improvement. Patience: %d/%d", patience_counter, cfg.training.patience)

            if epoch % cfg.training.save_every == 0:
                ckpt_path = f"checkpoints/wipose_epoch_{epoch:03d}.pt"
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ref_offsets": ref_offsets.cpu(),
                    "metrics": all_metrics,
                    "config": OmegaConf.to_container(cfg),
                }, ckpt_path)

            if patience_counter >= cfg.training.patience:
                log.info("Early stopping at epoch %d", epoch)
                break

            if cfg.training.dry_run:
                break

        log.info("Training complete. Best MPJPE: %.4f", best_mpjpe)
        mlflow.log_metric("best_mpjpe", best_mpjpe)


if __name__ == "__main__":
    train()
