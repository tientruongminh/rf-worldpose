"""
train_v2.py
-----------
Training script for model variants:
    - root_relative: Root-relative coordinate prediction (#5)
    - subcarrier_attn: Subcarrier-aware attention tokenizer (#4)

Imports infrastructure from transformer_train.py, overrides only
build_model_and_tokenizer and loss computation for the variants.

Config key: model.variant = "rootrel" | "subcarrier_attn"

Usage:
    python -m rfpose.training.train_v2 --config-name rootrel_eagle
    python -m rfpose.training.train_v2 --config-name subcarrier_attn_eagle
"""

import os
import time
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from rfpose.models.csi_tokenizer import CSITokenizer
from rfpose.models.transformer import CSITransformerPose
from rfpose.data.gold_npz_dataset import NUM_ACTIONS

from rfpose.training.transformer_train import (
    _is_ddp, _rank, _world_size, _is_main, _setup_ddp, _cleanup_ddp,
    _resolve_checkpoint_path,
    build_dataloaders, build_loss_fn, build_optimizer, build_scheduler,
    save_checkpoint, load_checkpoint,
    load_ssl_pretrained, _freeze_encoder, _export_onnx,
    eval_one_epoch,
)
from rfpose.utils.losses import MPJPE, PA_MPJPE, RFPoseLoss

log = logging.getLogger(__name__)


# Root joint indices for 13-joint skeleton: l_hip=7, r_hip=10
ROOT_JOINT_IDS_13 = (7, 10)


def _compute_root_gt(gt_coords: torch.Tensor, n_joints: int = 13) -> torch.Tensor:
    """Compute root (pelvis center) from ground truth coordinates.
    
    For 13 joints: root = mean(l_hip, r_hip)
    For 17 joints (COCO): root = mean(left_hip=11, right_hip=12)
    """
    if n_joints == 13:
        return (gt_coords[:, :, 7, :] + gt_coords[:, :, 10, :]) / 2.0
    elif n_joints == 17:
        return (gt_coords[:, :, 11, :] + gt_coords[:, :, 12, :]) / 2.0
    else:
        return gt_coords.mean(dim=2)


def build_model_and_tokenizer_v2(cfg: DictConfig):
    """Build model and tokenizer based on model.variant config."""
    variant = cfg.model.get("variant", "base")

    if variant == "rootrel":
        from rfpose.models.transformer_rootrel import CSITransformerPoseRootRel

        tokenizer = CSITokenizer(
            n_subcarriers=cfg.data.n_subcarriers,
            patch_size=cfg.model.patch_size,
            d_model=cfg.model.d_model,
            max_seq_len=cfg.data.window_size + 10,
            n_nodes=cfg.data.n_nodes,
            dropout=cfg.model.dropout,
        )

        model = CSITransformerPoseRootRel(
            n_patches=tokenizer.n_patches,
            d_model=cfg.model.d_model,
            spatial_heads=cfg.model.spatial_heads,
            temporal_heads=cfg.model.temporal_heads,
            n_spatial_layers=cfg.model.n_spatial_layers,
            n_temporal_layers=cfg.model.n_temporal_layers,
            n_decoder_layers=cfg.model.n_decoder_layers,
            n_decoder_temporal_layers=cfg.model.get("n_decoder_temporal_layers", 2),
            n_joints=cfg.data.n_joints,
            predict_3d=cfg.model.predict_3d,
            causal_temporal=cfg.model.get("causal_temporal", False),
            dropout=cfg.model.dropout,
            ffn_mult=cfg.model.ffn_mult,
            n_nodes=cfg.data.n_nodes,
            num_actions=cfg.model.get("num_actions", NUM_ACTIONS),
        )

        return tokenizer, model

    elif variant == "subcarrier_attn":
        from rfpose.models.csi_tokenizer_attn import CSITokenizerAttn

        n_tokens = cfg.model.get("n_tokens", cfg.data.n_subcarriers // cfg.model.patch_size)
        n_attn_heads = cfg.model.get("n_attn_heads", 4)

        tokenizer = CSITokenizerAttn(
            n_subcarriers=cfg.data.n_subcarriers,
            n_tokens=n_tokens,
            d_model=cfg.model.d_model,
            max_seq_len=cfg.data.window_size + 10,
            n_nodes=cfg.data.n_nodes,
            dropout=cfg.model.dropout,
            n_attn_heads=n_attn_heads,
        )

        model = CSITransformerPose(
            n_patches=tokenizer.n_patches,
            d_model=cfg.model.d_model,
            spatial_heads=cfg.model.spatial_heads,
            temporal_heads=cfg.model.temporal_heads,
            n_spatial_layers=cfg.model.n_spatial_layers,
            n_temporal_layers=cfg.model.n_temporal_layers,
            n_decoder_layers=cfg.model.n_decoder_layers,
            n_decoder_temporal_layers=cfg.model.get("n_decoder_temporal_layers", 2),
            n_joints=cfg.data.n_joints,
            predict_3d=cfg.model.predict_3d,
            causal_temporal=cfg.model.get("causal_temporal", False),
            dropout=cfg.model.dropout,
            ffn_mult=cfg.model.ffn_mult,
            n_nodes=cfg.data.n_nodes,
            num_actions=cfg.model.get("num_actions", NUM_ACTIONS),
        )

        return tokenizer, model

    elif variant == "gcn_rootrel":
        from rfpose.models.pose_decoder_gcn import CSITransformerPoseGCN

        tokenizer = CSITokenizer(
            n_subcarriers=cfg.data.n_subcarriers,
            patch_size=cfg.model.patch_size,
            d_model=cfg.model.d_model,
            max_seq_len=cfg.data.window_size + 10,
            n_nodes=cfg.data.n_nodes,
            dropout=cfg.model.dropout,
        )

        model = CSITransformerPoseGCN(
            n_patches=tokenizer.n_patches,
            d_model=cfg.model.d_model,
            spatial_heads=cfg.model.spatial_heads,
            temporal_heads=cfg.model.temporal_heads,
            n_spatial_layers=cfg.model.n_spatial_layers,
            n_temporal_layers=cfg.model.n_temporal_layers,
            n_gcn_layers=cfg.model.get("n_gcn_layers", 3),
            n_gcn_tf_layers=cfg.model.get("n_gcn_tf_layers", 3),
            n_joints=cfg.data.n_joints,
            predict_3d=cfg.model.predict_3d,
            causal_temporal=cfg.model.get("causal_temporal", False),
            dropout=cfg.model.dropout,
            ffn_mult=cfg.model.ffn_mult,
            n_nodes=cfg.data.n_nodes,
            num_actions=cfg.model.get("num_actions", NUM_ACTIONS),
        )

        return tokenizer, model

    elif variant == "metafi":
        from rfpose.models.metafi_baseline import MetaFiTokenizer, MetaFiModel

        tokenizer = MetaFiTokenizer(
            n_subcarriers=cfg.data.n_subcarriers,
            n_channels=2,
            d_model=cfg.model.d_model,
            max_seq_len=cfg.data.window_size + 10,
            dropout=cfg.model.dropout,
        )

        model = MetaFiModel(
            d_model=cfg.model.d_model,
            n_layers=cfg.model.get("n_encoder_layers", 4),
            n_heads=cfg.model.get("n_encoder_heads", 8),
            n_joints=cfg.data.n_joints,
            num_actions=cfg.model.get("num_actions", NUM_ACTIONS),
            dropout=cfg.model.dropout,
            ffn_mult=cfg.model.get("ffn_mult", 2),
        )

        return tokenizer, model

    else:
        raise ValueError(f"Unknown model variant: {variant}")


def load_ssl_pretrained_v2(tokenizer, model, ssl_checkpoint, freeze_encoder=False):
    """Load SSL pretrained encoder — handles both CSITokenizer and CSITokenizerAttn."""
    ckpt = torch.load(ssl_checkpoint, map_location="cpu", weights_only=True)
    log.info("Loading SSL pretrained encoder from %s (epoch=%s)", ssl_checkpoint, ckpt.get("epoch", "?"))

    tokenizer.load_state_dict(ckpt["tokenizer"], strict=False)
    model.spatial_encoder.load_state_dict(ckpt["spatial_encoder"], strict=False)
    model.temporal_encoder.load_state_dict(ckpt["temporal_encoder"], strict=False)

    if "spatial_to_temporal_norm" in ckpt:
        model.spatial_to_temporal_norm.load_state_dict(
            ckpt["spatial_to_temporal_norm"], strict=False,
        )

    log.info("SSL encoder loaded (strict=False). freeze_encoder=%s", freeze_encoder)

    if freeze_encoder:
        _freeze_encoder(tokenizer, model)


def train_one_epoch_v2(
    tokenizer, model, loader, optimizer, scheduler,
    loss_fn, device, scaler, cfg, epoch,
):
    """Training loop with root-relative loss support."""
    tokenizer.train()
    model.train()

    variant = cfg.model.get("variant", "base")
    is_rootrel = variant in ("rootrel", "gcn_rootrel")
    lambda_root = cfg.loss.get("lambda_root", 1.0)
    lambda_offset = cfg.loss.get("lambda_offset", 1.0)
    lambda_action = cfg.loss.get("lambda_action", 0.0)
    action_ce = nn.CrossEntropyLoss(ignore_index=-1) if lambda_action > 0 else None
    n_joints = cfg.data.n_joints

    # Per-joint weighting: upweight extremities (knees/ankles) that have 2x error
    joint_weights_list = cfg.loss.get("joint_weights", None)
    if joint_weights_list is not None:
        joint_weights = torch.tensor(joint_weights_list, dtype=torch.float32, device=device)
        joint_weights = joint_weights / joint_weights.mean()  # normalize to mean=1
    else:
        joint_weights = None

    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        csi = batch["csi"].to(device, non_blocking=True)
        gt_coords = batch["coords"].to(device, non_blocking=True)
        gt_vis = batch["vis"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=cfg.training.amp):
            tokens = tokenizer(csi)
            out = model(tokens)

            pred_dict = {"coords": out["coords"], "vis_logits": out["vis_logits"]}
            gt_dict = {"coords": gt_coords, "vis": gt_vis}
            loss, breakdown = loss_fn(pred_dict, gt_dict)

            # Root-relative losses
            if is_rootrel:
                gt_root = _compute_root_gt(gt_coords, n_joints)
                gt_offsets = gt_coords - gt_root.unsqueeze(2)

                root_loss = F.smooth_l1_loss(out["root"], gt_root)
                breakdown["loss_root"] = root_loss.item()
                loss = loss + lambda_root * root_loss

                vis_mask = gt_vis.unsqueeze(-1).expand_as(out["offsets"])
                pred_off_vis = out["offsets"] * vis_mask
                gt_off_vis = gt_offsets * vis_mask

                if joint_weights is not None:
                    # [J] -> [1, 1, J, 1] for broadcasting with [B, T, J, 3]
                    jw = joint_weights.view(1, 1, -1, 1).expand_as(pred_off_vis)
                    off_err = F.smooth_l1_loss(pred_off_vis, gt_off_vis, reduction="none")
                    off_loss = (off_err * jw * vis_mask).sum() / vis_mask.sum().clamp(min=1.0)
                else:
                    off_loss = F.smooth_l1_loss(
                        pred_off_vis, gt_off_vis, reduction="sum",
                    ) / vis_mask.sum().clamp(min=1.0)

                breakdown["loss_offset"] = off_loss.item()
                loss = loss + lambda_offset * off_loss

            # Action classification
            if action_ce is not None and "action_label" in batch:
                gt_action = batch["action_label"].to(device, non_blocking=True)
                a_mask = batch.get("action_mask", torch.ones_like(gt_action, dtype=torch.float32))
                a_mask = a_mask.to(device, non_blocking=True).bool()
                if a_mask.any():
                    action_loss = action_ce(out["action_logits"][a_mask], gt_action[a_mask])
                    loss = loss + lambda_action * action_loss
                    breakdown["loss_action"] = action_loss.item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(tokenizer.parameters()) + list(model.parameters()) + list(loss_fn.parameters()),
            max_norm=cfg.training.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k, v in breakdown.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

        if batch_idx % cfg.training.log_every == 0 and _is_main():
            lr = scheduler.get_last_lr()[0]
            extra = ""
            if is_rootrel:
                extra += f" root={breakdown.get('loss_root', 0):.4f}"
                extra += f" offset={breakdown.get('loss_offset', 0):.4f}"
            if "loss_action" in breakdown:
                extra += f" action={breakdown['loss_action']:.4f}"
            log.info(
                "Epoch %d [%d/%d] loss=%.4f%s lr=%.6f",
                epoch, batch_idx, len(loader), breakdown["loss_total"], extra, lr,
            )
            global_step = epoch * len(loader) + batch_idx
            step_metrics = {
                "step_loss": breakdown["loss_total"],
                "step_coord": breakdown.get("loss_coord", 0),
                "lr": lr,
            }
            if is_rootrel:
                step_metrics["step_root_loss"] = breakdown.get("loss_root", 0)
                step_metrics["step_offset_loss"] = breakdown.get("loss_offset", 0)
            mlflow.log_metrics(step_metrics, step=global_step)

        if cfg.training.dry_run:
            break

    return {k: v / n_batches for k, v in total_metrics.items()}


@hydra.main(config_path="../../configs", config_name="rootrel_eagle", version_base=None)
def train(cfg: DictConfig) -> None:
    _setup_ddp()
    rank = _rank()
    world = _world_size()

    if _is_main():
        log.info("=" * 60)
        log.info("RF-WorldPose Train V2 — variant=%s", cfg.model.get("variant", "base"))
        log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
        log.info("DDP: %d GPU(s)", world)
        log.info("=" * 60)

    if _is_ddp():
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    if _is_main():
        log.info("Device: %s", device)
        if device.type == "cuda":
            log.info("GPU: %s", torch.cuda.get_device_name(device))

    torch.manual_seed(cfg.training.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed + rank)

    train_loader, val_loader, n_train, n_val = build_dataloaders(cfg, device)
    if _is_main():
        log.info("Train: %d | Val: %d", n_train, n_val)

    tokenizer, model = build_model_and_tokenizer_v2(cfg)

    # Load pretrained weights
    ssl_ckpt = cfg.training.get("ssl_pretrained", "")
    pretrained_from = cfg.training.get("pretrained_from", "")
    freeze = cfg.training.get("freeze_encoder", False)

    if _is_main():
        ssl_ckpt = _resolve_checkpoint_path(ssl_ckpt)
        pretrained_from = _resolve_checkpoint_path(pretrained_from)
    if _is_ddp():
        obj_list = [ssl_ckpt, pretrained_from]
        dist.broadcast_object_list(obj_list, src=0)
        ssl_ckpt, pretrained_from = obj_list

    if ssl_ckpt and Path(ssl_ckpt).exists():
        load_ssl_pretrained_v2(tokenizer, model, ssl_ckpt, freeze_encoder=freeze)

    tokenizer = tokenizer.to(device)
    model = model.to(device)

    raw_model = model
    raw_tokenizer = tokenizer
    if _is_ddp():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)

    if _is_main():
        trainable_tok = sum(p.numel() for p in raw_tokenizer.parameters() if p.requires_grad)
        trainable_mod = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        log.info("Tokenizer params: %s (trainable)", f"{trainable_tok:,}")
        log.info("Model params:     %s (trainable)", f"{trainable_mod:,}")

    loss_fn = build_loss_fn(cfg).to(device)
    optimizer = build_optimizer(raw_model, raw_tokenizer, cfg, loss_fn=loss_fn)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = amp.GradScaler(device=str(device), enabled=cfg.training.amp)

    start_epoch = 0
    best_mpjpe = float("inf")

    if cfg.training.resume_from and Path(cfg.training.resume_from).exists():
        start_epoch, prev_metrics = load_checkpoint(
            cfg.training.resume_from, raw_tokenizer, raw_model, optimizer, scheduler,
        )
        best_mpjpe = prev_metrics.get("val_mpjpe", float("inf"))
        start_epoch += 1

    if _is_main():
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_ctx = mlflow.start_run(run_name=cfg.mlflow.run_name) if _is_main() else nullcontext()

    with run_ctx:
        if _is_main():
            mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))
            mlflow.log_params(OmegaConf.to_container(cfg.training, resolve=True))
            mlflow.log_params(OmegaConf.to_container(cfg.loss, resolve=True))
            mlflow.log_param("ddp_world_size", world)
            mlflow.log_param("variant", cfg.model.get("variant", "base"))

        patience_counter = 0

        for epoch in range(start_epoch, cfg.training.epochs):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            t0 = time.time()

            train_metrics = train_one_epoch_v2(
                tokenizer, model, train_loader,
                optimizer, scheduler, loss_fn,
                device, scaler, cfg, epoch,
            )

            val_metrics = eval_one_epoch(
                raw_tokenizer, raw_model, val_loader,
                loss_fn, device, cfg,
            )

            epoch_time = time.time() - t0

            if _is_main():
                all_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "epoch_time": epoch_time}
                mlflow.log_metrics(all_metrics, step=epoch)

                log.info(
                    "Epoch %03d/%d [%.1fs] train_loss=%.4f val_mpjpe=%.4f val_pa_mpjpe=%.4f",
                    epoch, cfg.training.epochs, epoch_time,
                    train_metrics["loss_total"],
                    val_metrics["val_mpjpe"],
                    val_metrics["val_pa_mpjpe"],
                )

                val_mpjpe = val_metrics["val_mpjpe"]
                if val_mpjpe < best_mpjpe:
                    best_mpjpe = val_mpjpe
                    patience_counter = 0
                    best_ckpt = "checkpoints/best.pt"
                    save_checkpoint(
                        epoch, raw_tokenizer, raw_model,
                        optimizer, scheduler, all_metrics, cfg, best_ckpt,
                    )
                    try:
                        mlflow.log_artifact(best_ckpt)
                    except Exception as e:
                        log.warning("Failed to upload artifact: %s", e)
                    log.info("  New best MPJPE: %.4f", best_mpjpe)
                else:
                    patience_counter += 1
                    log.info("  No improvement. Patience: %d/%d", patience_counter, cfg.training.patience)

                if epoch % cfg.training.save_every == 0:
                    save_checkpoint(
                        epoch, raw_tokenizer, raw_model, optimizer, scheduler,
                        all_metrics, cfg, f"checkpoints/epoch_{epoch:03d}.pt",
                    )

                if patience_counter >= cfg.training.patience:
                    log.info("Early stopping at epoch %d", epoch)
                    break

            if cfg.training.dry_run:
                if _is_main():
                    log.info("Dry run complete.")
                break

        if _is_main():
            log.info("Training complete. Best MPJPE: %.4f", best_mpjpe)
            mlflow.log_metric("best_mpjpe", best_mpjpe)

    _cleanup_ddp()


if __name__ == "__main__":
    train()
