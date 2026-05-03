from __future__ import annotations
import torch
from torch import nn

class CsiTokenizer(nn.Module):
    def __init__(self, channels: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(channels, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,T,S,C] -> [B,N,T,D]
        # Production models can use sparse subcarrier attention; this smoke-safe
        # implementation pools subcarriers to avoid O((N*T*S)^2) attention.
        z = self.proj(x).mean(dim=3)
        b, n, t, d = z.shape
        return z.reshape(b, n * t, d)

class RFGraphTransformer(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)

class RFWorldPose(nn.Module):
    """Initial RF-WorldPose model.

    Implements the production target shape now:
      CSI Tokenizer + RF Graph Transformer + pooled latent + heads.
    Later extensions plug in Neural RF Field, SMPL, DensePose, and LoRA adapters.
    """
    def __init__(
        self,
        num_nodes: int = 4,
        window_frames: int = 60,
        n_subcarriers: int = 56,
        channels: int = 2,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        num_classes: int = 6,
        num_keypoints: int = 17,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_frames = window_frames
        self.n_subcarriers = n_subcarriers
        self.channels = channels
        self.tokenizer = CsiTokenizer(channels, dim)
        self.node_embed = nn.Embedding(num_nodes, dim)
        self.subcarrier_embed = nn.Embedding(n_subcarriers, dim)
        self.time_embed = nn.Embedding(window_frames, dim)
        self.transformer = RFGraphTransformer(dim, depth, heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.action_head = nn.Linear(dim, num_classes)
        self.presence_head = nn.Linear(dim, 1)
        self.keypoint_head = nn.Linear(dim, num_keypoints * 3)

    def positional_bias(self, device: torch.device) -> torch.Tensor:
        n, t = self.num_nodes, self.window_frames
        node_ids = torch.arange(n, device=device).view(n, 1).expand(n, t).reshape(-1)
        time_ids = torch.arange(t, device=device).view(1, t).expand(n, t).reshape(-1)
        return self.node_embed(node_ids) + self.time_embed(time_ids)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(x)
        tokens = tokens + self.positional_bias(x.device).unsqueeze(0)
        z = self.transformer(tokens)
        pooled = self.norm(z.mean(dim=1))
        keypoints = self.keypoint_head(pooled).view(x.shape[0], -1, 3)
        return {
            "action_logits": self.action_head(pooled),
            "presence_logit": self.presence_head(pooled).squeeze(-1),
            "keypoints": keypoints,
            "embedding": pooled,
        }
