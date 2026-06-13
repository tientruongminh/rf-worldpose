"""Minimal DDP helpers (torchrun --standalone --nproc_per_node=N)."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_module(module: torch.nn.Module | None) -> torch.nn.Module | None:
    if module is None:
        return None
    return module.module if isinstance(module, DDP) else module


def wrap_ddp(module: torch.nn.Module, local_rank: int, world_size: int) -> torch.nn.Module:
    if world_size <= 1:
        return module
    return DDP(module, device_ids=[local_rank], output_device=local_rank)
