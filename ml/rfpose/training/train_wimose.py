"""Train Wi-Mose network with optional DDP (torchrun --nproc_per_node=N).

Usage (single GPU):
    python train_wimose.py --config-name wimose_mmfi17j_eagle

Usage (2 GPUs, DDP):
    torchrun --standalone --nproc_per_node=2 train_wimose.py \\
        --config-name wimose_mmfi17j_eagle

Key differences from CSIViT2DPose / WiPose training:
  - Model input: (B, 2, N_sub, T)  — CSI treated as a 2-D image
  - GT         : middle frame of the T-frame window → (B, J, 3)
  - Loss       : MSE + Huber(δ=0.75)  (no FK / quaternion)
  - Metric     : MPJPE  (mm if poses stored in mm, m if in m)
"""
from __future__ import annotations

import os
import logging
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rfpose.data.gold_npz_dataset import (
    GoldNpzDataset,
    NUM_ACTIONS,
    _SubsetGoldNpz,
    build_gold_train_val,
)
from rfpose.models.wimose_net import WiMoseNet, WiMoseLoss, uniformity_loss, compute_ref_offsets, diversity_metrics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean Per Joint Position Error (same units as input)."""
    return (pred - gt).norm(dim=-1).mean().item()


def _run_model(
    model: nn.Module,
    x: torch.Tensor,
    *,
    return_features: bool = False,
    with_action: bool = False,
) -> dict[str, torch.Tensor]:
    out = model(x, return_features=return_features, return_action=with_action)
    if isinstance(out, dict):
        return out
    if return_features:
        coords, feat = out
        return {"coords": coords, "features": feat}
    return {"coords": out}


@torch.no_grad()
def _compute_csi_stats(dataset, n_sample: int = 256, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel CSI mean/std estimated from a random sample of windows.

    Returns mean, std shaped (1, 2, 1, 1) ready to broadcast over (B, 2, N_sub, T).
    Deterministic given seed, so every DDP rank computes identical stats without
    needing a collective broadcast.
    """
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[: min(n_sample, n)].tolist()
    sums = torch.zeros(2, dtype=torch.float64)
    sqs = torch.zeros(2, dtype=torch.float64)
    count = 0
    for i in idx:
        csi = dataset[i]["csi"]            # (T, N_sub, 2)
        flat = csi.reshape(-1, 2).double()
        sums += flat.sum(dim=0)
        sqs += (flat * flat).sum(dim=0)
        count += flat.shape[0]
    mean = (sums / max(count, 1))
    var = (sqs / max(count, 1) - mean * mean).clamp(min=1e-12)
    std = var.sqrt()
    mean_t = mean.float().view(1, 2, 1, 1)
    std_t = std.float().view(1, 2, 1, 1)
    return mean_t, std_t


def _compute_action_class_weights(
    train_ds,
    num_actions: int,
) -> torch.Tensor:
    """Inverse-frequency class weights from the train split (sklearn-style)."""
    counts = torch.zeros(num_actions, dtype=torch.float64)
    if isinstance(train_ds, _SubsetGoldNpz):
        base = train_ds.base
        for subset_idx in train_ds.indices:
            entry = base.entries[subset_idx]
            y = np.load(entry["y_path"])
            lab = int(y["action_label"][entry["index"]]) if "action_label" in y else 0
            y.close()
            if 0 <= lab < num_actions:
                counts[lab] += 1
    elif isinstance(train_ds, GoldNpzDataset):
        for entry in train_ds.entries:
            y = np.load(entry["y_path"])
            lab = int(y["action_label"][entry["index"]]) if "action_label" in y else 0
            y.close()
            if 0 <= lab < num_actions:
                counts[lab] += 1
    else:
        for i in range(len(train_ds)):
            lab = int(train_ds[i]["action_label"].item())
            if 0 <= lab < num_actions:
                counts[lab] += 1

    n_samples = counts.sum().item()
    n_present = int((counts > 0).sum().item())
    if n_samples <= 0 or n_present == 0:
        return torch.ones(num_actions, dtype=torch.float32)

    weights = torch.zeros(num_actions, dtype=torch.float64)
    present = counts > 0
    weights[present] = n_samples / (n_present * counts[present])
    weights[~present] = weights[present].mean() if present.any() else 1.0
    return weights.float()


def _prepare_batch(
    batch: dict,
    device: torch.device,
    csi_mean: torch.Tensor | None,
    csi_std: torch.Tensor | None,
    root_joint: int,
    center_pose: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build normalized CSI image and root-relative GT for a batch.

    Returns (x, gt, mask):
        x   : (B, 2, N_sub, T)  — channel-normalized CSI image
        gt  : (B, J, 3)         — middle-frame pose, optionally root-relative
        mask: (B,)
    """
    csi    = batch["csi"].to(device, non_blocking=True)        # (B, T, N_sub, 2)
    coords = batch["coords"].to(device, non_blocking=True)     # (B, T, J, 3)
    mask   = batch["pose_mask"].to(device, non_blocking=True)  # (B,)

    x = csi.permute(0, 3, 2, 1).contiguous()                   # (B, 2, N_sub, T)
    if csi_mean is not None and csi_std is not None:
        x = (x - csi_mean) / csi_std

    T = coords.shape[1]
    gt = coords[:, T // 2, :, :]                               # (B, J, 3)
    if center_pose and 0 <= root_joint < gt.shape[1]:
        gt = gt - gt[:, root_joint : root_joint + 1, :]        # root-relative

    return x, gt, mask


def _setup_ddp() -> tuple[int, int, int]:
    """Initialise NCCL process group if launched with torchrun.

    Returns: (rank, local_rank, world_size)
    """
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def _cleanup_ddp(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Per-epoch loops
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: WiMoseLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    rank: int,
    csi_mean: torch.Tensor | None = None,
    csi_std: torch.Tensor | None = None,
    root_joint: int = 0,
    center_pose: bool = True,
    log_every: int = 50,
    lambda_unif: float = 0.0,
    lambda_action: float = 0.0,
    action_ce: nn.CrossEntropyLoss | None = None,
) -> tuple[float, float, float]:
    """Returns (avg_loss, avg_mpjpe, avg_action_acc). action_acc=0 if disabled."""
    model.train()
    total_loss = 0.0
    total_mpjpe = 0.0
    action_correct = 0
    action_total = 0
    n = 0
    t_batch = time.time()
    use_unif = lambda_unif > 0.0
    use_action = lambda_action > 0.0 and action_ce is not None

    for batch_idx, batch in enumerate(loader):
        x, gt, mask = _prepare_batch(batch, device, csi_mean, csi_std, root_joint, center_pose)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = _run_model(
                model, x,
                return_features=use_unif,
                with_action=use_action,
            )
            pred = out["coords"]
            loss = criterion(pred, gt, mask)
            if use_unif:
                loss = loss + lambda_unif * uniformity_loss(out["features"])
            if use_action and action_ce is not None and "action_logits" in out:
                gt_action = batch["action_label"].to(device, non_blocking=True)
                a_mask = batch.get(
                    "action_mask",
                    torch.ones_like(gt_action, dtype=torch.float32),
                ).to(device, non_blocking=True).bool()
                if a_mask.any():
                    logits = out["action_logits"][a_mask]
                    labels = gt_action[a_mask]
                    loss = loss + lambda_action * action_ce(logits, labels)
                    action_correct += (logits.argmax(dim=-1) == labels).sum().item()
                    action_total += a_mask.sum().item()

        if not torch.isfinite(loss):
            log.warning("rank=%d non-finite loss=%.4f — batch skipped", rank, loss.item())
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        B = x.size(0)
        total_loss  += loss.item() * B
        total_mpjpe += _mpjpe(pred.detach(), gt.detach()) * B
        n            += B

        if rank == 0 and log_every > 0 and (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - t_batch
            log.info(
                "  [batch %d]  loss=%.4f  mpjpe=%.4f  %.1fs",
                batch_idx + 1, loss.item(), _mpjpe(pred.detach(), gt.detach()), elapsed,
            )
            t_batch = time.time()

    if n == 0:
        return float("nan"), float("nan"), 0.0
    action_acc = action_correct / action_total if action_total > 0 else 0.0
    return total_loss / n, total_mpjpe / n, action_acc


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: WiMoseLoss,
    device: torch.device,
    use_amp: bool,
    csi_mean: torch.Tensor | None = None,
    csi_std: torch.Tensor | None = None,
    root_joint: int = 0,
    center_pose: bool = True,
) -> tuple[float, float, float, float, float]:
    model.eval()
    total_loss  = 0.0
    total_mpjpe = 0.0
    total_std_ratio = 0.0
    total_spread_ratio = 0.0
    action_correct = 0
    action_total = 0
    n = 0
    with_action = any(
        hasattr(m, "action_head") and m.action_head is not None
        for m in ([model.module] if isinstance(model, DDP) else [model])
    )

    for batch in loader:
        x, gt, mask = _prepare_batch(batch, device, csi_mean, csi_std, root_joint, center_pose)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = _run_model(model, x, with_action=with_action)
            pred = out["coords"]
            loss = criterion(pred, gt, mask)

        if with_action and "action_logits" in out and "action_label" in batch:
            gt_action = batch["action_label"].to(device, non_blocking=True)
            a_mask = batch.get(
                "action_mask",
                torch.ones_like(gt_action, dtype=torch.float32),
            ).to(device, non_blocking=True).bool()
            if a_mask.any():
                logits = out["action_logits"][a_mask]
                labels = gt_action[a_mask]
                action_correct += (logits.argmax(dim=-1) == labels).sum().item()
                action_total += a_mask.sum().item()

        if torch.isfinite(loss):
            B = x.size(0)
            total_loss  += loss.item() * B
            total_mpjpe += _mpjpe(pred, gt) * B
            dm = diversity_metrics(pred.detach(), gt.detach())
            total_std_ratio += dm["std_ratio"] * B
            total_spread_ratio += dm["spread_ratio"] * B
            n            += B

    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0.0
    action_acc = action_correct / action_total if action_total > 0 else 0.0
    return (
        total_loss / n, total_mpjpe / n,
        total_std_ratio / n, total_spread_ratio / n,
        action_acc,
    )


# ---------------------------------------------------------------------------
# Main training loop (Hydra entry point)
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="wimose_mmfi17j_eagle", version_base=None)
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size = _setup_ddp()
    is_main = (rank == 0)

    # ── logging ──────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s [rank%(levelname)s] %(message)s",
    )
    if is_main:
        log.info("Wi-Mose | world_size=%d\n%s", world_size, OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.training.seed + rank)
    device = torch.device("cuda", local_rank)

    # ── data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds = build_gold_train_val(
        cfg.data.gold_dir,
        datasets=list(cfg.data.datasets),
        augment=cfg.data.augment,
        require_pose=True,
        val_splits=("val",),  # keep `test` truly held out for final eval
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) \
        if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False) \
        if world_size > 1 else None

    nw = cfg.training.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=nw,
        pin_memory=(nw > 0),
        persistent_workers=(nw > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,
        sampler=val_sampler,
        shuffle=False,
        num_workers=nw,
        pin_memory=(nw > 0),
        persistent_workers=(nw > 0),
    )

    # ── model + DDP wrap ─────────────────────────────────────────────────────
    use_fk_head = cfg.model.get("use_fk_head", False)
    num_actions = int(cfg.model.get("num_actions", 0))
    model = WiMoseNet(
        n_joints=cfg.data.n_joints,
        in_channels=2,
        use_gcn_head=cfg.model.get("use_gcn_head", False),
        use_fk_head=use_fk_head,
        gcn_dim=cfg.model.get("gcn_dim", 256),
        gcn_layers=cfg.model.get("gcn_layers", 3),
        num_actions=num_actions,
    ).to(device)

    pretrained_from = cfg.training.get("pretrained_from", "")
    if pretrained_from and Path(pretrained_from).exists():
        ckpt = torch.load(pretrained_from, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main:
            log.info(
                "Loaded pretrained pose weights from %s (missing=%d unexpected=%d)",
                pretrained_from, len(missing), len(unexpected),
            )

    if cfg.training.get("freeze_backbone", False):
        for name, param in model.named_parameters():
            if not name.startswith("action_head"):
                param.requires_grad = False
        if is_main:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info("freeze_backbone=True — trainable params=%s", f"{trainable:,}")

    if use_fk_head:
        ref_offsets = compute_ref_offsets(
            train_ds,
            n_joints=cfg.data.n_joints,
            root_joint=cfg.data.get("root_joint", 0),
            n_sample=512,
            seed=cfg.training.seed,
        ).to(device)
        model.ref_offsets.copy_(ref_offsets)
        if is_main:
            log.info("FK ref_offsets computed from train split (n_joints=%d)", cfg.data.n_joints)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        log.info("WiMoseNet  params = %s", f"{n_params:,}")

    # ── preprocessing: CSI normalization + root-relative pose centering ───────
    normalize_csi = cfg.data.get("normalize_csi", True)
    center_pose   = cfg.data.get("center_pose", True)
    root_joint    = cfg.data.get("root_joint", 0)  # mmfi: Pelvis=0, wipose: Neck=1
    csi_mean = csi_std = None
    if normalize_csi:
        csi_mean, csi_std = _compute_csi_stats(train_ds, n_sample=256, seed=cfg.training.seed)
        csi_mean = csi_mean.to(device)
        csi_std  = csi_std.to(device)
        if is_main:
            log.info(
                "CSI norm  mean=%s  std=%s | center_pose=%s root_joint=%d",
                csi_mean.flatten().tolist(), csi_std.flatten().tolist(),
                center_pose, root_joint,
            )

    # ── loss / optim / scheduler ─────────────────────────────────────────────
    criterion = WiMoseLoss(
        delta=cfg.loss.get("huber_delta", 0.75),
        lambda_bone=cfg.loss.get("lambda_bone", 0.5),
        lambda_sym=cfg.loss.get("lambda_sym", 0.1),
        lambda_div=cfg.loss.get("lambda_div", 0.05),
        lambda_collapse=cfg.loss.get("lambda_collapse", 0.0),
        lambda_spread=cfg.loss.get("lambda_spread", 0.0),
    ).to(device)
    lambda_unif_val = cfg.loss.get("lambda_unif", 0.0)
    lambda_action_val = float(cfg.loss.get("lambda_action", 0.0))
    action_ce = None
    if lambda_action_val > 0.0 and num_actions > 0:
        action_weight = None
        if cfg.loss.get("action_class_weights", False):
            action_weight = _compute_action_class_weights(train_ds, num_actions).to(device)
            if is_main:
                top_w = sorted(
                    ((i, action_weight[i].item()) for i in range(num_actions) if action_weight[i] > 0),
                    key=lambda t: t[1],
                    reverse=True,
                )[:5]
                log.info("Action class weights enabled (sample): %s", top_w)
        action_ce = nn.CrossEntropyLoss(weight=action_weight)
    if is_main:
        log.info(
            "Loss: weighted(MSE+Huber δ=%.2f) + bone×%.2f + sym×%.2f + div×%.2f "
            "+ collapse×%.2f + spread×%.2f + unif×%.4f + action×%.2f (weighted_ce=%s) | fk_head=%s",
            cfg.loss.get("huber_delta", 0.75),
            cfg.loss.get("lambda_bone", 0.5),
            cfg.loss.get("lambda_sym", 0.1),
            cfg.loss.get("lambda_div", 0.05),
            cfg.loss.get("lambda_collapse", 0.0),
            cfg.loss.get("lambda_spread", 0.0),
            lambda_unif_val,
            lambda_action_val,
            bool(cfg.loss.get("action_class_weights", False)),
            use_fk_head,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.get("weight_decay", 0.0),
    )

    warmup_epochs = cfg.training.get("warmup_epochs", 3)
    def _warmup_lr(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_lr)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs - warmup_epochs,
        eta_min=cfg.training.lr * 0.01,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.get("amp", True))

    # ── checkpoint dir ───────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    if is_main:
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning("Cannot create checkpoint dir %s: %s — using /tmp fallback", ckpt_dir, e)
            ckpt_dir = Path("/tmp/wimose_ckpt")
            ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow ── (rank-0 only; barrier afterwards so all ranks enter together)
    mlflow_active = False
    if is_main:
        try:
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
            mlflow.set_experiment(cfg.mlflow.experiment_name)
            mlflow.start_run(run_name=cfg.mlflow.run_name)
            mlflow.log_params({
                "n_joints":    cfg.data.n_joints,
                "datasets":    str(list(cfg.data.datasets)),
                "epochs":      cfg.training.epochs,
                "batch_size":  cfg.training.batch_size,
                "world_size":  world_size,
                "lr":          cfg.training.lr,
                "huber_delta": cfg.loss.get("huber_delta", 0.75),
            })
            mlflow_active = True
            log.info("MLflow run started: %s", cfg.mlflow.run_name)
        except Exception as e:
            log.warning("MLflow setup failed (continuing without): %s", e)

    # Synchronise all ranks before entering the training loop
    if world_size > 1:
        dist.barrier()

    # ── train loop ───────────────────────────────────────────────────────────
    early_stop_metric = cfg.training.get("early_stop_metric", "val_mpjpe")
    stop_on_action = early_stop_metric == "val_action_acc"
    best_val_mpjpe = float("inf")
    best_val_action_acc = 0.0
    patience_left  = cfg.training.get("patience", 25)

    for epoch in range(cfg.training.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        tr_loss, tr_mpjpe, tr_action_acc = _train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            cfg.training.get("amp", True), rank,
            csi_mean=csi_mean, csi_std=csi_std,
            root_joint=root_joint, center_pose=center_pose,
            lambda_unif=cfg.loss.get("lambda_unif", 0.0),
            lambda_action=lambda_action_val,
            action_ce=action_ce,
        )

        va_loss, va_mpjpe, va_std_ratio, va_spread_ratio, va_action_acc = _val_epoch(
            model, val_loader, criterion, device, cfg.training.get("amp", True),
            csi_mean=csi_mean, csi_std=csi_std,
            root_joint=root_joint, center_pose=center_pose,
        )

        # Scheduler step
        if epoch < warmup_epochs:
            warmup_sched.step()
        else:
            cosine_sched.step()

        elapsed = time.time() - t0

        if is_main:
            log.info(
                "Epoch %3d/%d  tr_loss=%.4f tr_mpjpe=%.4f  "
                "val_loss=%.4f val_mpjpe=%.4f  std_ratio=%.3f spread_ratio=%.3f  "
                "tr_action_acc=%.3f val_action_acc=%.3f  lr=%.2e  %.1fs",
                epoch + 1, cfg.training.epochs,
                tr_loss, tr_mpjpe, va_loss, va_mpjpe,
                va_std_ratio, va_spread_ratio,
                tr_action_acc, va_action_acc,
                optimizer.param_groups[0]["lr"], elapsed,
            )
            if mlflow_active:
                try:
                    mlflow.log_metrics({
                        "train_loss":   tr_loss,
                        "train_mpjpe":  tr_mpjpe,
                        "val_loss":     va_loss,
                        "val_mpjpe":    va_mpjpe,
                        "val_std_ratio": va_std_ratio,
                        "val_spread_ratio": va_spread_ratio,
                        "train_action_acc": tr_action_acc,
                        "val_action_acc": va_action_acc,
                        "lr":           optimizer.param_groups[0]["lr"],
                    }, step=epoch)
                except Exception as e:
                    log.warning("mlflow log_metrics failed: %s", e)

            # Save periodic checkpoint
            save_every = cfg.training.get("save_every", 10)
            raw_model  = model.module if isinstance(model, DDP) else model
            if (epoch + 1) % save_every == 0:
                ep_ckpt = ckpt_dir / f"epoch_{epoch+1:03d}.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mpjpe": va_mpjpe,
                }, ep_ckpt)
                if mlflow_active:
                    try:
                        mlflow.log_artifact(str(ep_ckpt))
                    except Exception as e:
                        log.warning("mlflow artifact upload failed: %s", e)

            def _save_checkpoint(path: Path, extra: dict | None = None) -> None:
                payload = {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "val_mpjpe": va_mpjpe,
                    "val_action_acc": va_action_acc,
                    "n_joints":  cfg.data.n_joints,
                    "datasets":  list(cfg.data.datasets),
                    "csi_mean":  None if csi_mean is None else csi_mean.cpu(),
                    "csi_std":   None if csi_std is None else csi_std.cpu(),
                    "center_pose": center_pose,
                    "root_joint":  root_joint,
                    "use_fk_head": use_fk_head,
                    "use_gcn_head": cfg.model.get("use_gcn_head", False) and not use_fk_head,
                    "num_actions": num_actions,
                    "ref_offsets": raw_model.ref_offsets.cpu() if use_fk_head else None,
                }
                if extra:
                    payload.update(extra)
                torch.save(payload, path)

            improved = False
            if va_mpjpe < best_val_mpjpe:
                best_val_mpjpe = va_mpjpe
                _save_checkpoint(ckpt_dir / "best.pt")
                log.info("  ✓ new best val_mpjpe=%.4f → best.pt", best_val_mpjpe)
                if not stop_on_action:
                    improved = True

            if va_action_acc > best_val_action_acc:
                best_val_action_acc = va_action_acc
                _save_checkpoint(ckpt_dir / "best_action.pt")
                log.info("  ✓ new best val_action_acc=%.3f → best_action.pt", best_val_action_acc)
                if stop_on_action:
                    improved = True

            if improved:
                patience_left = cfg.training.get("patience", 25)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log.info(
                        "Early stopping at epoch %d (metric=%s)",
                        epoch + 1, early_stop_metric,
                    )
                    break

    if is_main:
        log.info(
            "Training complete. Best val_mpjpe=%.4f  best val_action_acc=%.3f",
            best_val_mpjpe, best_val_action_acc,
        )
        if mlflow_active:
            try:
                mlflow.log_metric("best_val_mpjpe", best_val_mpjpe)
                mlflow.log_metric("best_val_action_acc", best_val_action_acc)
                mlflow.end_run()
            except Exception as e:
                log.warning("mlflow end_run failed: %s", e)

    _cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
