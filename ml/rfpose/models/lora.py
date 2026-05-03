from __future__ import annotations
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for p in self.base.parameters():
            p.requires_grad = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale

def apply_lora_to_heads(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> nn.Module:
    for name in ["action_head", "presence_head", "keypoint_head"]:
        layer = getattr(model, name, None)
        if isinstance(layer, nn.Linear):
            setattr(model, name, LoRALinear(layer, rank=rank, alpha=alpha))
    return model

def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
