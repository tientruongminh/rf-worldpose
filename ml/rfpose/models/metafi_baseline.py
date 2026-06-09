"""
metafi_baseline.py
------------------
MetaFi++ baseline model (WPFormer) adapted for our CSI format and pipeline.

Reference: Zhou et al., "MetaFi++: WiFi-Enabled Transformer-Based Human Pose
Estimation for Metaverse Avatar Simulation", IEEE IoT Journal 2023.

Architecture:
    MetaFiTokenizer:  RunningNorm → flatten → Linear → pos embedding
    MetaFiModel:      TransformerEncoder → per-frame pose head + attention-pool action head

Much simpler than our CSITransformerPose (~2M vs ~15M params):
    - No separate spatial/temporal encoders
    - No cross-attention decoder with joint queries
    - Single TransformerEncoder over temporal dimension
    - Direct MLP regression per frame
"""

import torch
import torch.nn as nn


class MetaFiTokenizer(nn.Module):
    """Flatten CSI → project → positional encoding (MetaFi++ style)."""

    def __init__(
        self,
        n_subcarriers: int = 270,
        n_channels: int = 2,
        d_model: int = 256,
        max_seq_len: int = 70,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = n_subcarriers * n_channels
        self.proj = nn.Linear(self.input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.n_patches = 1  # compatibility with param counting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N_sub, 2) → (B, T, d_model)"""
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1)
        x = self.proj(x) + self.pos[:, :T]
        return self.dropout(x)


class MetaFiModel(nn.Module):
    """MetaFi++ Transformer encoder + per-frame pose regression.

    Differences from original MetaFi++:
        - Outputs per-frame 3D pose (not single-frame 2D)
        - Includes visibility and action heads for our multi-task loss
        - Adapted for 270 subcarriers (vs original 114×3=342)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_joints: int = 13,
        num_actions: int = 28,
        dropout: float = 0.2,
        ffn_mult: int = 2,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.pose_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_joints * 3),
        )
        self.vis_head = nn.Linear(d_model, n_joints)

        self.action_attn = nn.Linear(d_model, 1)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_actions),
        )
        self.n_joints = n_joints

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """tokens: (B, T, d_model) from MetaFiTokenizer."""
        h = self.encoder(tokens)  # (B, T, d_model)
        B, T, _ = h.shape

        coords = self.pose_head(h).reshape(B, T, self.n_joints, 3)
        vis_logits = self.vis_head(h)

        w = torch.softmax(self.action_attn(h), dim=1)
        pooled = (h * w).sum(dim=1)
        action_logits = self.action_head(pooled)

        return {
            "coords": coords,
            "vis_logits": vis_logits,
            "action_logits": action_logits,
        }
