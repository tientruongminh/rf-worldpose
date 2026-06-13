"""WiMAE pretraining — Phase 1 of DT-Pose-style pipeline.

Trains a Masked Auto-Encoder on WiFi CSI images (no pose labels needed).
The encoder learns discriminative, diverse representations because it must
reconstruct 80% of masked signal tokens from only 20% visible context.

After pretraining the encoder weights are saved to ``encoder.pt`` inside
``checkpoint_dir``.  The diffusion / fine-tuning stage loads this encoder.

Usage (single GPU):
    python pretrain_mae.py --config-name mae_pretrain_mmfi_eagle

Usage (4-GPU DDP):
    torchrun --standalone --nproc_per_node=4 pretrain_mae.py \\
        --config-name mae_pretrain_mmfi_eagle
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import hydra
import math
import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rfpose.data.gold_npz_dataset import build_gold_train_val
from rfpose.models.wimose_mae import WiMoseMAE
from rfpose.training.train_wimose import _compute_csi_stats, _setup_ddp, _cleanup_ddp

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csi_batch(batch: dict, device: torch.device, csi_mean, csi_std):
    """Extract and normalise CSI image only (no pose labels needed)."""
    csi = batch["csi"].to(device, non_blocking=True)     # (B, T, N_sub, 2)
    x   = csi.permute(0, 3, 2, 1).contiguous()           # (B, 2, N_sub, T)
    if csi_mean is not None:
        x = (x - csi_mean) / csi_std
    return x


# ---------------------------------------------------------------------------
# Per-epoch loops
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    rank: int,
    csi_mean,
    csi_std,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        x = _csi_batch(batch, device, csi_mean, csi_std)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out  = model(x)
            loss = out["loss"]

        if not torch.isfinite(loss):
            log.warning("rank=%d non-finite loss at step %d — skipped", rank, i)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        B = x.size(0)
        total_loss += loss.item() * B
        n += B

        if rank == 0 and log_every > 0 and (i + 1) % log_every == 0:
            log.info("  [%d]  recon_loss=%.5f  %.1fs", i + 1, loss.item(), time.time() - t0)
            t0 = time.time()

    return total_loss / max(n, 1)


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    csi_mean,
    csi_std,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = _csi_batch(batch, device, csi_mean, csi_std)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
        if torch.isfinite(out["loss"]):
            B = x.size(0)
            total_loss += out["loss"].item() * B
            n += B
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="mae_pretrain_mmfi_eagle", version_base=None)
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size = _setup_ddp()
    is_main = rank == 0
    device  = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # ── dataset ──────────────────────────────────────────────────────────────
    train_ds, val_ds = build_gold_train_val(
        gold_dir=cfg.data.gold_dir,
        datasets=list(cfg.data.datasets),
        augment=cfg.data.get("augment", False),
        require_pose=cfg.data.get("require_pose", False),  # MAE doesn't need pose
        val_splits=("val",),
    )

    if is_main:
        log.info("MAE dataset: train=%d  val=%d", len(train_ds), len(val_ds))

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=cfg.training.get("num_workers", 4),
        pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size * 2,
        sampler=val_sampler, shuffle=False,
        num_workers=cfg.training.get("num_workers", 4),
        pin_memory=True, drop_last=False, persistent_workers=True,
    )

    # ── CSI normalisation ────────────────────────────────────────────────────
    csi_mean = csi_std = None
    if cfg.data.get("normalize_csi", True):
        csi_mean, csi_std = _compute_csi_stats(train_ds, n_sample=256, seed=42)
        csi_mean = csi_mean.to(device)
        csi_std  = csi_std.to(device)
        if is_main:
            log.info("CSI stats: mean=%s  std=%s", csi_mean.flatten().tolist(), csi_std.flatten().tolist())

    # ── model ────────────────────────────────────────────────────────────────
    model = WiMoseMAE(
        mask_ratio=cfg.model.get("mask_ratio", 0.80),
        in_channels=2,
        img_h=cfg.data.n_subcarriers,
        img_w=cfg.data.window_size,
        patch_h=cfg.model.get("patch_h", 18),
        patch_w=cfg.model.get("patch_w", 6),
        encoder_dim=cfg.model.get("encoder_dim", 384),
        encoder_layers=cfg.model.get("encoder_layers", 6),
        encoder_heads=cfg.model.get("encoder_heads", 6),
        decoder_dim=cfg.model.get("decoder_dim", 256),
        decoder_layers=cfg.model.get("decoder_layers", 4),
        decoder_heads=cfg.model.get("decoder_heads", 4),
        dropout=cfg.model.get("dropout", 0.1),
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        log.info("WiMoseMAE params = %s", f"{n_params:,}")

    # ── optimiser + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.get("weight_decay", 0.05),
    )
    warmup   = cfg.training.get("warmup_epochs", 10)
    total_ep = cfg.training.epochs

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
                "task":          "mae_pretrain",
                "mask_ratio":    cfg.model.get("mask_ratio", 0.80),
                "encoder_dim":   cfg.model.get("encoder_dim", 384),
                "encoder_layers":cfg.model.get("encoder_layers", 6),
                "epochs":        total_ep,
                "batch_size":    cfg.training.batch_size,
                "lr":            cfg.training.lr,
                "world_size":    world_size,
            })
            mlflow_active = True
        except Exception as e:
            log.warning("MLflow failed: %s", e)

    if world_size > 1:
        dist.barrier()

    # ── training loop ────────────────────────────────────────────────────────
    best_val = float("inf")
    for epoch in range(total_ep):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss = _train_epoch(
            model, train_loader, optimizer, scaler, device,
            cfg.training.get("amp", True), rank, csi_mean, csi_std,
        )
        va_loss = _val_epoch(model, val_loader, device, cfg.training.get("amp", True), csi_mean, csi_std)
        scheduler.step()

        if is_main:
            lr_now = optimizer.param_groups[0]["lr"]
            log.info("Epoch %3d  tr_recon=%.5f  va_recon=%.5f  lr=%.2e", epoch, tr_loss, va_loss, lr_now)
            if mlflow_active:
                mlflow.log_metrics({"tr_recon": tr_loss, "va_recon": va_loss, "lr": lr_now}, step=epoch)

            # Save best encoder
            if va_loss < best_val:
                best_val = va_loss
                enc_module = model.module.encoder if world_size > 1 else model.encoder
                torch.save({
                    "epoch":      epoch,
                    "val_recon":  va_loss,
                    "encoder":    enc_module.state_dict(),
                    "csi_mean":   csi_mean,
                    "csi_std":    csi_std,
                    "embed_dim":  cfg.model.get("encoder_dim", 384),
                    "n_layers":   cfg.model.get("encoder_layers", 6),
                    "n_heads":    cfg.model.get("encoder_heads", 6),
                    "patch_h":    cfg.model.get("patch_h", 18),
                    "patch_w":    cfg.model.get("patch_w", 6),
                }, ckpt_dir / "encoder_best.pt")
                log.info("  ✓ New best encoder  val_recon=%.5f → saved", va_loss)

            # Periodic full checkpoint
            if (epoch + 1) % cfg.training.get("save_every", 10) == 0:
                full_model = model.module if world_size > 1 else model
                torch.save({
                    "epoch":  epoch,
                    "model":  full_model.state_dict(),
                    "optim":  optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                }, ckpt_dir / f"mae_epoch{epoch:04d}.pt")

    if is_main and mlflow_active:
        mlflow.end_run()

    _cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
