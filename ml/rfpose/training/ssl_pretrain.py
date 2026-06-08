"""
ssl_pretrain.py — Self-supervised pretraining for CSI encoder.

Phase 1 of training pipeline:
  CSI (no labels) → CSITokenizer + SpatialEncoder + TemporalEncoder
  SSL tasks: Masked CSI Reconstruction + Temporal Contrastive

Output: checkpoints/csi_encoder_pretrained.pt
  Contains: tokenizer + spatial_encoder + temporal_encoder state_dict

Usage:
  python -m rfpose.training.ssl_pretrain
  python -m rfpose.training.ssl_pretrain ssl.mask_ratio=0.4 training.epochs=30
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
from torch.utils.data import DataLoader, random_split

import hydra
from omegaconf import DictConfig, OmegaConf

import mlflow

from rfpose.models.csi_tokenizer import CSITokenizer
from rfpose.models.transformer import SpatialEncoder, TemporalEncoder

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _mlflow_upload_ckpt(ckpt_path: Path, artifact_subdir: str) -> None:
    """Upload checkpoint to MLflow artifacts (best-effort)."""
    if not mlflow.active_run() or not ckpt_path.exists():
        return
    try:
        mlflow.log_artifact(str(ckpt_path), artifact_path=artifact_subdir)
        log.info("MLflow artifact: %s -> artifacts/%s", ckpt_path.name, artifact_subdir)
    except Exception as e:
        log.warning("MLflow artifact upload failed (%s): %s", ckpt_path.name, e)


class GoldCsiOnlyDataset(torch.utils.data.Dataset):
    """Load pre-windowed CSI from Gold x.npy files — no labels, SSL only."""

    def __init__(self, gold_dir: str | Path, datasets: list[str] | None = None):
        self.entries: list[dict] = []
        gold = Path(gold_dir)
        for ds_dir in sorted(gold.iterdir()):
            if not ds_dir.is_dir():
                continue
            if datasets and ds_dir.name not in datasets:
                continue
            x_path = ds_dir / "x.npy"
            if not x_path.exists():
                continue
            import numpy as _np
            n = _np.load(str(x_path), mmap_mode="r").shape[0]
            for i in range(n):
                self.entries.append({"x_path": str(x_path), "index": i})
        log.info("GoldCsiOnlyDataset: %d windows from %s", len(self.entries), gold)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.entries[idx]
        import numpy as _np
        x = _np.load(e["x_path"], mmap_mode="r")
        win = _np.array(x[e["index"]], dtype=_np.float32)  # (2, T, N_sub) writable copy
        _np.nan_to_num(win, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        for ch in range(win.shape[0]):
            mu = win[ch].mean()
            std = win[ch].std() + 1e-8
            win[ch] = (win[ch] - mu) / std
        csi = torch.from_numpy(win.transpose(1, 2, 0))  # (T, N_sub, 2)
        return {"csi": csi}


class MaskedReconstructionHead(nn.Module):
    """Reconstruct masked patches from encoder features."""

    def __init__(self, d_model: int, patch_size: int):
        super().__init__()
        patch_dim = patch_size * 2
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_dim),
        )
        self.patch_size = patch_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)  # (B, T, N_patches, patch_dim)


class TemporalProjectionHead(nn.Module):
    """Project CLS-like pooled feature for contrastive loss."""

    def __init__(self, d_model: int, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class SSLEncoder(nn.Module):
    """Wraps tokenizer + spatial + temporal encoders for SSL."""

    def __init__(
        self,
        tokenizer: CSITokenizer,
        spatial_encoder: SpatialEncoder,
        temporal_encoder: TemporalEncoder,
        d_model: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder
        self.spatial_to_temporal_norm = nn.LayerNorm(d_model)

    def forward(
        self, csi: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        csi: (B, T, N_sub, 2)
        mask: (B, T, N_patches) bool — True = masked
        Returns: (B, T, N_patches, d_model)
        """
        tokens = self.tokenizer(csi)  # (B, T, N_patches, D)

        if mask is not None:
            B, T, N, D = tokens.shape
            mask_token = torch.zeros(1, 1, 1, D, device=tokens.device)
            tokens = tokens * (~mask).unsqueeze(-1).float() + mask_token * mask.unsqueeze(-1).float()

        spatial_feat = self.spatial_encoder(tokens)
        spatial_feat = self.spatial_to_temporal_norm(spatial_feat)
        temporal_feat = self.temporal_encoder(spatial_feat)
        return temporal_feat  # (B, T, N_patches, D)


def generate_mask(B: int, T: int, N: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """Random patch mask. Returns (B, T, N) bool tensor."""
    n_mask = int(T * N * mask_ratio)
    mask = torch.zeros(B, T * N, dtype=torch.bool, device=device)
    for i in range(B):
        idx = torch.randperm(T * N, device=device)[:n_mask]
        mask[i, idx] = True
    return mask.reshape(B, T, N)


def masked_reconstruction_loss(
    pred: torch.Tensor,
    target_csi: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    pred: (B, T, N_patches, patch_dim) — reconstructed patches
    target_csi: (B, T, N_sub, 2) — original CSI
    mask: (B, T, N_patches) — True = was masked
    """
    B, T, N_sub, C = target_csi.shape
    N_patches = N_sub // patch_size

    # Reshape target to patches: (B, T, N_patches, patch_size * 2)
    target = target_csi.reshape(B, T, N_patches, patch_size, C)
    target = target.reshape(B, T, N_patches, patch_size * C)

    # Only compute loss on masked positions
    mask_expanded = mask.unsqueeze(-1).expand_as(pred)
    pred_masked = pred[mask_expanded].reshape(-1, pred.shape[-1])
    target_masked = target[mask_expanded].reshape(-1, target.shape[-1])

    if pred_masked.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    return F.smooth_l1_loss(pred_masked, target_masked)


def temporal_contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE between two augmented views. z1, z2: (B, D) normalized."""
    B = z1.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z1.device)

    sim = torch.mm(z1, z2.T) / temperature  # (B, B)
    labels = torch.arange(B, device=z1.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


def augment_csi(csi: torch.Tensor) -> torch.Tensor:
    """Light augmentation for contrastive view."""
    aug = csi.clone()
    if torch.rand(1) < 0.7:
        aug = aug + torch.randn_like(aug) * 0.03
    if torch.rand(1) < 0.5:
        T = aug.shape[1]
        t_mask = torch.randint(0, T, (max(1, T // 10),))
        aug[:, t_mask] = 0
    if torch.rand(1) < 0.5:
        N = aug.shape[2]
        n_mask = torch.randint(0, N, (max(1, N // 10),))
        aug[:, :, n_mask] = 0
    return aug


@hydra.main(config_path="../../configs", config_name="ssl_pretrain", version_base=None)
def pretrain(cfg: DictConfig) -> None:
    log.info(f"\n{'='*60}")
    log.info("RF-WorldPose SSL Pretraining (Masked Reconstruction + Contrastive)")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"{'='*60}")

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.training.seed)

    # Dataset — Gold (pre-windowed) or Silver (raw)
    data_format = cfg.data.get("format", "silver")
    if data_format == "gold_npz":
        dataset = GoldCsiOnlyDataset(
            cfg.data.gold_dir,
            datasets=list(cfg.data.datasets) if cfg.data.get("datasets") else None,
        )
    else:
        from rfpose.data.silver_csi_dataset import SilverCsiDataset
        dataset = SilverCsiDataset(
            cfg.data.silver_dir,
            unified_dir=cfg.data.get("unified_dir"),
            window_size=cfg.data.window_size,
            min_timesteps=cfg.data.min_timesteps,
            n_padded=cfg.data.get("n_padded"),
        )

    n_val = max(1, int(len(dataset) * cfg.data.val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )

    nw = cfg.training.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size * 2, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"),
    )
    log.info(f"Train: {n_train} | Val: {n_val}")

    # Build encoder
    tokenizer = CSITokenizer(
        n_subcarriers=cfg.data.n_subcarriers,
        patch_size=cfg.model.patch_size,
        d_model=cfg.model.d_model,
        max_seq_len=cfg.data.window_size + 10,
        n_nodes=1,
        dropout=cfg.model.dropout,
    )
    n_patches = tokenizer.n_patches

    spatial_encoder = SpatialEncoder(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.spatial_heads,
        n_layers=cfg.model.n_spatial_layers,
        ffn_mult=cfg.model.ffn_mult,
        dropout=cfg.model.dropout,
    )
    temporal_encoder = TemporalEncoder(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.temporal_heads,
        n_layers=cfg.model.n_temporal_layers,
        ffn_mult=cfg.model.ffn_mult,
        dropout=cfg.model.dropout,
    )

    encoder = SSLEncoder(tokenizer, spatial_encoder, temporal_encoder, cfg.model.d_model).to(device)
    recon_head = MaskedReconstructionHead(cfg.model.d_model, cfg.model.patch_size).to(device)
    contrast_head = TemporalProjectionHead(cfg.model.d_model, cfg.ssl.proj_dim).to(device)

    all_params = list(encoder.parameters()) + list(recon_head.parameters()) + list(contrast_head.parameters())
    total_params = sum(p.numel() for p in all_params if p.requires_grad)
    log.info(f"SSL Encoder params: {total_params:,}")

    optimizer = torch.optim.AdamW(all_params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    total_steps = cfg.training.epochs * len(train_loader)
    warmup_steps = cfg.training.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = amp.GradScaler(device=cfg.training.device, enabled=cfg.training.amp)

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # MLflow
    mlflow_cfg = cfg.get("mlflow") or {}
    mlflow_enabled = bool(mlflow_cfg.get("tracking_uri"))
    mlflow_log_batch = mlflow_cfg.get("log_batch_metrics", True)
    mlflow_log_ckpt = mlflow_cfg.get("log_checkpoints", True)
    mlflow_log_every = mlflow_cfg.get("log_every") or cfg.training.log_every

    if mlflow_enabled:
        try:
            mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "rf-worldpose-ssl"))
            mlflow.start_run(run_name=mlflow_cfg.get("run_name", "ssl-pretrain"))
            mlflow.log_params({
                "ssl.mask_ratio": cfg.ssl.mask_ratio,
                "ssl.lambda_recon": cfg.ssl.lambda_recon,
                "ssl.lambda_contrast": cfg.ssl.lambda_contrast,
                "model.d_model": cfg.model.d_model,
                "model.n_spatial_layers": cfg.model.n_spatial_layers,
                "model.n_temporal_layers": cfg.model.n_temporal_layers,
                "training.epochs": cfg.training.epochs,
                "training.batch_size": cfg.training.batch_size,
                "training.lr": cfg.training.lr,
                "data.n_train": n_train,
                "data.n_val": n_val,
                "total_params": total_params,
                "mlflow.log_batch_metrics": mlflow_log_batch,
                "mlflow.log_checkpoints": mlflow_log_ckpt,
            })
            log.info("MLflow tracking enabled: %s", mlflow_cfg.tracking_uri)
        except Exception as e:
            log.warning("MLflow init failed: %s", e)
            mlflow_enabled = False

    for epoch in range(cfg.training.epochs):
        t0 = time.time()

        # --- Train ---
        encoder.train()
        recon_head.train()
        contrast_head.train()
        train_recon, train_contrast, train_total = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            csi = batch["csi"].to(device, non_blocking=True)
            B, T, N_sub, C = csi.shape
            N = n_patches

            # Masked reconstruction
            mask = generate_mask(B, T, N, cfg.ssl.mask_ratio, device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type, enabled=cfg.training.amp):
                features = encoder(csi, mask=mask)
                recon_pred = recon_head(features)
                loss_recon = masked_reconstruction_loss(recon_pred, csi, mask, cfg.model.patch_size)

                # Contrastive: two augmented views → pool → project → InfoNCE
                csi_aug1 = augment_csi(csi)
                csi_aug2 = augment_csi(csi)
                feat1 = encoder(csi_aug1).mean(dim=(1, 2))  # (B, D)
                feat2 = encoder(csi_aug2).mean(dim=(1, 2))
                z1 = contrast_head(feat1)
                z2 = contrast_head(feat2)
                loss_contrast = temporal_contrastive_loss(z1, z2, cfg.ssl.temperature)

                loss = cfg.ssl.lambda_recon * loss_recon + cfg.ssl.lambda_contrast * loss_contrast

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_recon += loss_recon.item()
            train_contrast += loss_contrast.item()
            train_total += loss.item()
            n_batches += 1

            global_step = epoch * len(train_loader) + n_batches

            if n_batches % cfg.training.log_every == 0:
                log.info(
                    f"Epoch {epoch} [{n_batches}/{len(train_loader)}] "
                    f"recon={loss_recon.item():.4f} contrast={loss_contrast.item():.4f} "
                    f"total={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.6f}"
                )

            if (
                mlflow_enabled
                and mlflow_log_batch
                and mlflow.active_run()
                and n_batches % mlflow_log_every == 0
            ):
                try:
                    mlflow.log_metrics({
                        "batch/recon_loss": loss_recon.item(),
                        "batch/contrast_loss": loss_contrast.item(),
                        "batch/total_loss": loss.item(),
                        "batch/lr": scheduler.get_last_lr()[0],
                    }, step=global_step)
                except Exception as e:
                    log.warning("MLflow batch log failed: %s", e)

            if cfg.training.get("dry_run", False):
                break

        train_recon /= max(n_batches, 1)
        train_contrast /= max(n_batches, 1)
        train_total /= max(n_batches, 1)

        # --- Validate ---
        encoder.eval()
        recon_head.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                csi = batch["csi"].to(device)
                B, T, N_sub, C = csi.shape
                mask = generate_mask(B, T, n_patches, cfg.ssl.mask_ratio, device)

                with amp.autocast(device_type=device.type, enabled=cfg.training.amp):
                    features = encoder(csi, mask=mask)
                    recon_pred = recon_head(features)
                    loss_r = masked_reconstruction_loss(recon_pred, csi, mask, cfg.model.patch_size)
                    val_loss += loss_r.item()

                n_val_batches += 1
                if cfg.training.get("dry_run", False):
                    break

        val_loss /= max(n_val_batches, 1)
        epoch_time = time.time() - t0

        log.info(
            f"Epoch {epoch:03d}/{cfg.training.epochs} [{epoch_time:.1f}s] "
            f"train_recon={train_recon:.4f} train_contrast={train_contrast:.4f} "
            f"train_total={train_total:.4f} val_recon={val_loss:.4f}"
        )

        epoch_step = (epoch + 1) * len(train_loader)
        if mlflow_enabled and mlflow.active_run():
            try:
                mlflow.log_metrics({
                    "epoch/train_recon": train_recon,
                    "epoch/train_contrast": train_contrast,
                    "epoch/train_total": train_total,
                    "epoch/val_recon": val_loss,
                    "epoch/lr": scheduler.get_last_lr()[0],
                    "epoch/time_sec": epoch_time,
                }, step=epoch_step)
            except Exception as e:
                log.warning("MLflow epoch log failed: %s", e)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = checkpoint_dir / "csi_encoder_pretrained.pt"
            torch.save({
                "epoch": epoch,
                "tokenizer": encoder.tokenizer.state_dict(),
                "spatial_encoder": encoder.spatial_encoder.state_dict(),
                "temporal_encoder": encoder.temporal_encoder.state_dict(),
                "spatial_to_temporal_norm": encoder.spatial_to_temporal_norm.state_dict(),
                "val_recon_loss": val_loss,
                "config": OmegaConf.to_container(cfg),
            }, save_path)
            log.info(f"Best model saved: {save_path} (val_recon={val_loss:.4f})")
            if mlflow_log_ckpt:
                _mlflow_upload_ckpt(
                    save_path,
                    f"checkpoints/best/epoch_{epoch:03d}_val_{val_loss:.4f}",
                )

        if (epoch + 1) % cfg.training.save_every == 0:
            epoch_ckpt = checkpoint_dir / f"ssl_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "tokenizer": encoder.tokenizer.state_dict(),
                "spatial_encoder": encoder.spatial_encoder.state_dict(),
                "temporal_encoder": encoder.temporal_encoder.state_dict(),
                "spatial_to_temporal_norm": encoder.spatial_to_temporal_norm.state_dict(),
                "val_recon_loss": val_loss,
            }, epoch_ckpt)
            if mlflow_log_ckpt:
                _mlflow_upload_ckpt(
                    epoch_ckpt,
                    f"checkpoints/periodic/epoch_{epoch:03d}",
                )

    log.info(f"SSL pretraining complete. Best val_recon={best_val_loss:.4f}")

    if mlflow_enabled and mlflow.active_run():
        try:
            mlflow.log_metric("best_val_recon", best_val_loss, step=cfg.training.epochs * len(train_loader))
            best_ckpt = checkpoint_dir / "csi_encoder_pretrained.pt"
            if mlflow_log_ckpt and best_ckpt.exists():
                _mlflow_upload_ckpt(best_ckpt, "checkpoints/final")
        except Exception as e:
            log.warning("MLflow final log failed: %s", e)
        mlflow.end_run()


if __name__ == "__main__":
    pretrain()
