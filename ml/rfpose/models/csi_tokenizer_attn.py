"""
csi_tokenizer_attn.py
---------------------
Subcarrier-Aware Attention Tokenizer variant of CSITokenizer.

Key idea: Replace fixed subcarrier patching (group 6 adjacent subcarriers,
project via Linear) with learnable cross-attention queries that attend to
ALL subcarriers simultaneously.

Benefits:
    - Data-driven subcarrier grouping: each token learns which frequency
      bands carry the most information for pose estimation
    - Body-part-specific frequency patterns: different tokens can specialize
      in different body parts by attending to different subcarrier groups
    - Adaptive to different environments: attention patterns can shift
      based on room geometry and multipath propagation
    - Attention weights are interpretable (which subcarriers → which tokens)

Output shape is compatible with the base Transformer model.
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

from rfpose.models.csi_tokenizer import (
    CSIRunningNorm,
    TemporalPositionalEncoding,
    MultiNodeFusion,
)


class SubcarrierAttentionEmbed(nn.Module):
    """
    Learnable query tokens attend to ALL subcarriers via cross-attention.

    n_tokens queries × n_subcarriers keys/values → n_tokens output embeddings
    per time step.
    """

    def __init__(
        self,
        n_subcarriers: int = 114,
        n_tokens: int = 19,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens

        self.subcarrier_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
        )

        self.subcarrier_pos = nn.Parameter(
            torch.zeros(1, n_subcarriers, d_model),
        )
        nn.init.trunc_normal_(self.subcarrier_pos, std=0.02)

        self.queries = nn.Parameter(torch.randn(n_tokens, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def n_patches(self) -> int:
        return self.n_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_sub, 2)
        returns: (B, T, n_tokens, d_model)
        """
        B, T, N, C = x.shape

        kv = self.subcarrier_proj(x)               # (B, T, N_sub, d_model)
        kv = kv + self.subcarrier_pos[:, :N, :]
        kv = rearrange(kv, "b t n d -> (b t) n d")

        q = self.queries.unsqueeze(0).expand(B * T, -1, -1)

        attended, _ = self.cross_attn(q, kv, kv)
        attended = self.norm(attended + q)
        attended = attended + self.ffn(attended)

        attended = rearrange(attended, "(b t) k d -> b t k d", b=B, t=T)
        return self.dropout(attended)


class CSITokenizerAttn(nn.Module):
    """
    Full tokenization pipeline using SubcarrierAttentionEmbed.

    Luồng:
        raw CSI (B, T, N_sub, 2)
           -> CSIRunningNorm               (normalize per-subcarrier)
           -> SubcarrierAttentionEmbed     (B, T, n_tokens, d_model)
           -> TemporalPositionalEnc        (+ temporal position info)
           -> output: (B, T, n_tokens, d_model)

    Drop-in replacement for CSITokenizer — same output shape when
    n_tokens == n_subcarriers // patch_size.
    """

    def __init__(
        self,
        n_subcarriers: int = 114,
        n_tokens: int = 19,
        d_model: int = 256,
        max_seq_len: int = 500,
        n_nodes: int = 1,
        dropout: float = 0.1,
        n_attn_heads: int = 4,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_patches = n_tokens

        if n_nodes > 1:
            self.node_fusion = MultiNodeFusion(n_nodes, n_subcarriers, d_model)
        else:
            self.node_fusion = None

        self.norm = CSIRunningNorm(n_subcarriers)
        self.embed = SubcarrierAttentionEmbed(
            n_subcarriers=n_subcarriers,
            n_tokens=n_tokens,
            d_model=d_model,
            n_heads=n_attn_heads,
            dropout=dropout,
        )
        self.temporal_pe = TemporalPositionalEncoding(d_model, max_seq_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_sub, 2)               — single node (4D)
           hoặc (B, n_nodes, T, N_sub, 2) — multi-node (5D)
        returns: (B, T, n_tokens, d_model)
        """
        if self.node_fusion is not None:
            assert x.ndim == 5, (
                f"Multi-node requires 5D Tensor [B, n_nodes, T, N_sub, 2], "
                f"got shape: {tuple(x.shape)}"
            )
            x = self.node_fusion(x)

        x = self.norm(x)            # (B, T, N_sub, 2)
        x = self.embed(x)           # (B, T, n_tokens, d_model)
        x = self.temporal_pe(x)     # (B, T, n_tokens, d_model)
        return x

    def flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b t p d -> b (t p) d")

    @property
    def output_shape_info(self) -> dict:
        return {
            "shape": "(B, T, n_tokens, d_model)",
            "flattened": "(B, T*n_tokens, d_model)",
            "n_patches": self.n_patches,
        }
