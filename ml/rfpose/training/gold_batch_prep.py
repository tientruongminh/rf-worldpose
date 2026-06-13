"""Shared Gold NPZ batch preprocessing for transformer-style trainers.

Matches Wi-Mose training pipeline:
  - per-channel CSI z-score (amp / phase)
  - root-relative pose coordinates (pelvis joint 0 for H36M)
"""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_csi_stats(
    dataset,
    n_sample: int = 256,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-channel CSI mean/std from a random sample of windows.

    Returns mean, std shaped (1, 1, 1, 2) for (B, T, N_sub, 2).
    """
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[: min(n_sample, n)].tolist()
    sums = torch.zeros(2, dtype=torch.float64)
    sqs = torch.zeros(2, dtype=torch.float64)
    count = 0
    for i in idx:
        csi = dataset[i]["csi"]  # (T, N_sub, 2)
        flat = csi.reshape(-1, 2).double()
        sums += flat.sum(dim=0)
        sqs += (flat * flat).sum(dim=0)
        count += flat.shape[0]
    mean = sums / max(count, 1)
    var = (sqs / max(count, 1) - mean * mean).clamp(min=1e-12)
    std = var.sqrt()
    mean_t = mean.float().view(1, 1, 1, 2)
    std_t = std.float().view(1, 1, 1, 2)
    return mean_t, std_t


def prepare_transformer_batch(
    batch: dict,
    device: torch.device,
    csi_mean: torch.Tensor | None,
    csi_std: torch.Tensor | None,
    root_joint: int,
    center_pose: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize CSI and optionally root-center pose for (B, T, J, 3) targets."""
    csi = batch["csi"].to(device, non_blocking=True)
    coords = batch["coords"].to(device, non_blocking=True)
    vis = batch["vis"].to(device, non_blocking=True)

    if csi_mean is not None and csi_std is not None:
        csi = (csi - csi_mean) / csi_std

    if center_pose and 0 <= root_joint < coords.shape[2]:
        root = coords[:, :, root_joint : root_joint + 1, :]
        coords = coords - root

    return csi, coords, vis
