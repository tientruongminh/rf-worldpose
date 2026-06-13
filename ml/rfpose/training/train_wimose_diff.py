"""Train WiMose Diffusion Decoder — Phase 2 (after MAE pretraining).

Two-stage usage:
  Stage A — pretrain: run pretrain_mae.py, get encoder_best.pt
  Stage B — fine-tune: this script, optionally loads encoder_best.pt

Usage (4-GPU DDP):
    torchrun --standalone --nproc_per_node=4 train_wimose_diff.py \\
        --config-name wimose_mmfi17j_diff_eagle \\
        training.pretrained_encoder=/path/to/encoder_best.pt

Usage without pretrained encoder (train from scratch, slower):
    torchrun --standalone --nproc_per_node=4 train_wimose_diff.py \\
        --config-name wimose_mmfi17j_diff_eagle

At inference, call model.sample(csi, schedule, n_hypotheses=5) to get
5 diverse pose hypotheses; MPJPE is evaluated on the best-of-N hypothesis.
"""
from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rfpose.data.gold_npz_dataset import build_gold_train_val
from rfpose.models.wimose_diffusion import WiMoseDiffNet, LinearNoiseSchedule
from rfpose.training.train_wimose import (
    _compute_csi_stats, _prepare_batch, _setup_ddp, _cleanup_ddp,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return (pred - gt).norm(dim=-1).mean().item()


def _best_of_n_mpjpe(
    hypotheses: torch.Tensor,  # (B, N, J, 3)
    gt: torch.Tensor,          # (B, J, 3)
) -> float:
    """Best-of-N MPJPE: for each sample take the hypothesis closest to GT."""
    gt_exp = gt.unsqueeze(1)                                   # (B, 1, J, 3)
    per_sample = (hypotheses - gt_exp).norm(dim=-1).mean(-1)   # (B, N) per-joint mean
    best = per_sample.min(dim=1).values                        # (B,) best hypothesis
    return best.mean().item()


# ---------------------------------------------------------------------------
# Per-epoch loops
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: LinearNoiseSchedule,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    rank: int,
    csi_mean, csi_std,
    root_joint: int,
    center_pose: bool,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        x, gt, mask = _prepare_batch(batch, device, csi_mean, csi_std, root_joint, center_pose)
        B = x.size(0)

        # Sample random timesteps
        t = torch.randint(0, schedule.T, (B,), device=device)

        # Forward diffusion: corrupt clean pose
        x_t, noise = schedule.q_sample(gt, t)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            noise_pred = model(x, x_t, t)                  # (B, J, 3)
            # Standard DDPM MSE on predicted vs actual noise
            loss = nn.functional.mse_loss(noise_pred, noise)

        if not torch.isfinite(loss):
            log.warning("rank=%d non-finite loss step=%d — skipped", rank, i)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * B
        n += B

        if rank == 0 and log_every > 0 and (i + 1) % log_every == 0:
            log.info("  [%d]  diff_loss=%.5f  %.1fs", i + 1, loss.item(), time.time() - t0)
            t0 = time.time()

    return total_loss / max(n, 1)


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: LinearNoiseSchedule,
    device: torch.device,
    use_amp: bool,
    csi_mean, csi_std,
    root_joint: int,
    center_pose: bool,
    n_ddim_steps: int = 20,
    n_hypotheses: int = 1,
) -> tuple[float, float]:
    """Returns (avg_diff_loss, avg_mpjpe)."""
    model.eval()
    total_loss  = 0.0
    total_mpjpe = 0.0
    n = 0

    # Unwrap DDP for sampling
    bare = model.module if hasattr(model, "module") else model

    for batch in loader:
        x, gt, mask = _prepare_batch(batch, device, csi_mean, csi_std, root_joint, center_pose)
        B = x.size(0)

        # ── DDPM training loss (random timesteps) ─────────────────────────
        t = torch.randint(0, schedule.T, (B,), device=device)
        x_t, noise = schedule.q_sample(gt, t)
        with torch.amp.autocast("cuda", enabled=use_amp):
            noise_pred = model(x, x_t, t)
            loss = nn.functional.mse_loss(noise_pred, noise)

        if torch.isfinite(loss):
            total_loss += loss.item() * B
            n += B

        # ── DDIM sampling for MPJPE ────────────────────────────────────────
        if n_hypotheses > 1:
            poses = bare.sample(x, schedule, n_steps=n_ddim_steps,
                                n_hypotheses=n_hypotheses, eta=0.0)
            poses = poses.view(B, n_hypotheses, -1, 3)  # (B, H, J, 3)
            total_mpjpe += _best_of_n_mpjpe(poses, gt) * B
        else:
            poses = bare.sample(x, schedule, n_steps=n_ddim_steps,
                                n_hypotheses=1, eta=0.0)  # (B, J, 3)
            total_mpjpe += _mpjpe(poses, gt) * B

    return total_loss / max(n, 1), total_mpjpe / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../../configs",
    config_name="wimose_mmfi17j_diff_eagle",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size = _setup_ddp()
    is_main = rank == 0
    device  = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # ── dataset ──────────────────────────────────────────────────────────────
    train_ds, val_ds = build_gold_train_val(
        gold_dir=cfg.data.gold_dir,
        datasets=list(cfg.data.datasets),
        augment=cfg.data.get("augment", True),
        require_pose=True,
        val_splits=("val",),
    )

    if is_main:
        log.info("Diffusion dataset: train=%d  val=%d", len(train_ds), len(val_ds))

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=cfg.training.get("num_workers", 4),
        pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=cfg.training.get("num_workers", 4),
        pin_memory=True, drop_last=False, persistent_workers=True,
    )

    # ── preprocessing ────────────────────────────────────────────────────────
    normalize_csi = cfg.data.get("normalize_csi", True)
    center_pose   = cfg.data.get("center_pose", True)
    root_joint    = cfg.data.get("root_joint", 0)
    csi_mean = csi_std = None
    if normalize_csi:
        csi_mean, csi_std = _compute_csi_stats(train_ds, n_sample=256, seed=42)
        csi_mean = csi_mean.to(device)
        csi_std  = csi_std.to(device)

    # ── noise schedule ───────────────────────────────────────────────────────
    T = cfg.model.get("T", 1000)
    schedule = LinearNoiseSchedule(
        T=T,
        beta_start=cfg.model.get("beta_start", 1e-4),
        beta_end=cfg.model.get("beta_end", 0.02),
    ).to(device)

    # ── model ────────────────────────────────────────────────────────────────
    wifi_dim = cfg.model.get("encoder_dim", 384)
    model = WiMoseDiffNet(
        n_joints=cfg.data.n_joints,
        wifi_dim=wifi_dim,
        dit_cfg={
            "d_model":    cfg.model.get("dit_d_model", 256),
            "n_layers":   cfg.model.get("dit_n_layers", 8),
            "n_heads":    cfg.model.get("dit_n_heads", 8),
            "t_embed_dim":cfg.model.get("t_embed_dim", 256),
        },
    ).to(device)

    # ── load pretrained encoder weights ─────────────────────────────────────
    pretrained_enc = cfg.training.get("pretrained_encoder", None)
    if pretrained_enc and Path(pretrained_enc).exists():
        ckpt = torch.load(pretrained_enc, map_location="cpu")
        missing, unexpected = model.encoder.load_state_dict(
            ckpt["encoder"], strict=False
        )
        if is_main:
            log.info(
                "Loaded pretrained MAE encoder: missing=%d unexpected=%d",
                len(missing), len(unexpected),
            )
    elif is_main:
        log.info("No pretrained encoder — training encoder from scratch")

    # ── optional encoder freeze ───────────────────────────────────────────
    freeze_enc_epochs = cfg.training.get("freeze_encoder_epochs", 0)
    if freeze_enc_epochs > 0 and is_main:
        log.info("Encoder will be frozen for first %d epochs", freeze_enc_epochs)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        log.info("WiMoseDiffNet params = %s", f"{n_params:,}")

    # ── optimiser + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.get("weight_decay", 1e-4),
    )
    total_ep = cfg.training.epochs
    warmup   = cfg.training.get("warmup_epochs", 5)

    def _lr_fn(ep: int) -> float:
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(total_ep - warmup, 1)
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.01)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_fn)
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.training.get("amp", True))

    # ── checkpoint dir ───────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_active = False
    if is_main:
        try:
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
            mlflow.set_experiment(cfg.mlflow.experiment_name)
            mlflow.start_run(run_name=cfg.mlflow.run_name)
            mlflow.log_params({
                "task":             "diffusion",
                "T":                T,
                "dit_d_model":      cfg.model.get("dit_d_model", 256),
                "dit_n_layers":     cfg.model.get("dit_n_layers", 8),
                "n_hypotheses_val": cfg.training.get("n_hypotheses_val", 1),
                "epochs":           total_ep,
                "lr":               cfg.training.lr,
                "pretrained_enc":   bool(pretrained_enc),
            })
            mlflow_active = True
        except Exception as e:
            log.warning("MLflow failed: %s", e)

    if world_size > 1:
        dist.barrier()

    # ── training loop ────────────────────────────────────────────────────────
    best_val_mpjpe = float("inf")
    n_hyp_val      = cfg.training.get("n_hypotheses_val", 1)
    n_ddim_steps   = cfg.training.get("n_ddim_steps", 20)
    patience_left  = cfg.training.get("patience", 30)

    for epoch in range(total_ep):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Unfreeze encoder after warmup
        bare_model = model.module if hasattr(model, "module") else model
        if epoch == freeze_enc_epochs and freeze_enc_epochs > 0:
            for p in bare_model.encoder.parameters():
                p.requires_grad_(True)
            if is_main:
                log.info("Epoch %d: encoder unfrozen", epoch)
        elif epoch < freeze_enc_epochs:
            for p in bare_model.encoder.parameters():
                p.requires_grad_(False)

        tr_loss = _train_epoch(
            model, train_loader, schedule, optimizer, scaler, device,
            cfg.training.get("amp", True), rank,
            csi_mean, csi_std, root_joint, center_pose,
        )

        va_loss, va_mpjpe = _val_epoch(
            model, val_loader, schedule, device, cfg.training.get("amp", True),
            csi_mean, csi_std, root_joint, center_pose,
            n_ddim_steps=n_ddim_steps,
            n_hypotheses=n_hyp_val,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        if is_main:
            log.info(
                "Epoch %3d  tr_diff=%.5f  va_diff=%.5f  va_mpjpe=%.4f  lr=%.2e",
                epoch, tr_loss, va_loss, va_mpjpe, lr_now,
            )
            if mlflow_active:
                mlflow.log_metrics({
                    "tr_diff_loss": tr_loss,
                    "va_diff_loss": va_loss,
                    "va_mpjpe":     va_mpjpe,
                    "lr":           lr_now,
                }, step=epoch)

            # Save best checkpoint
            if va_mpjpe < best_val_mpjpe:
                best_val_mpjpe = va_mpjpe
                patience_left  = cfg.training.get("patience", 30)
                torch.save({
                    "epoch":          epoch,
                    "val_mpjpe":      va_mpjpe,
                    "model":          bare_model.state_dict(),
                    "csi_mean":       csi_mean,
                    "csi_std":        csi_std,
                    "center_pose":    center_pose,
                    "root_joint":     root_joint,
                    "n_joints":       cfg.data.n_joints,
                    "T":              T,
                    "dit_d_model":    cfg.model.get("dit_d_model", 256),
                    "dit_n_layers":   cfg.model.get("dit_n_layers", 8),
                    "encoder_dim":    wifi_dim,
                }, ckpt_dir / "best.pt")
                log.info("  ✓ New best  val_mpjpe=%.4f → saved", va_mpjpe)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log.info("Early stop at epoch %d", epoch)
                    break

            if (epoch + 1) % cfg.training.get("save_every", 10) == 0:
                torch.save({
                    "epoch":  epoch,
                    "model":  bare_model.state_dict(),
                    "optim":  optimizer.state_dict(),
                }, ckpt_dir / f"diff_epoch{epoch:04d}.pt")

    if is_main and mlflow_active:
        mlflow.end_run()

    _cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
