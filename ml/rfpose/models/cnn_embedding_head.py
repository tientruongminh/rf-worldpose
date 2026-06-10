"""
cnn_embedding_head.py
---------------------
Frozen SSL CSI encoder + temporal CNN prediction head.
"""

import torch
import torch.nn as nn

from rfpose.models.csi_tokenizer import CSITokenizer
from rfpose.models.transformer import SpatialEncoder, TemporalEncoder


class TemporalCNNHead(nn.Module):
    """Temporal Conv1d head over SSL encoder embeddings."""

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        n_joints: int = 17,
        num_actions: int = 13,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("cnn_kernel_size must be odd to preserve temporal length")
        if n_layers < 1:
            raise ValueError("cnn_layers must be at least 1")

        layers: list[nn.Module] = []
        in_dim = d_model
        padding = kernel_size // 2
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.temporal_cnn = nn.Sequential(*layers)
        self.n_joints = n_joints
        self.coord_head = nn.Linear(hidden_dim, n_joints * 3)
        self.vis_head = nn.Linear(hidden_dim, n_joints)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (B, T, D)
        """
        x = x.transpose(1, 2)       # (B, D, T)
        x = self.temporal_cnn(x)
        x = x.transpose(1, 2)       # (B, T, H)

        coords = self.coord_head(x).reshape(x.shape[0], x.shape[1], self.n_joints, 3)
        vis_logits = self.vis_head(x)
        action_logits = self.action_head(x.mean(dim=1))

        return {
            "coords": coords,
            "vis_logits": vis_logits,
            "action_logits": action_logits,
        }


class CSIEncoderCNNModel(nn.Module):
    """Raw CSI -> SSL encoder embeddings -> temporal CNN predictions."""

    def __init__(
        self,
        tokenizer: CSITokenizer,
        spatial_encoder: SpatialEncoder,
        temporal_encoder: TemporalEncoder,
        spatial_to_temporal_norm: nn.LayerNorm,
        cnn_head: TemporalCNNHead,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder
        self.spatial_to_temporal_norm = spatial_to_temporal_norm
        self.cnn_head = cnn_head

    def forward(self, csi: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(csi)                    # (B, T, P, D)
        spatial_feat = self.spatial_encoder(tokens)     # (B, T, P, D)
        pooled = spatial_feat.mean(dim=2)               # (B, T, D)
        pooled = self.spatial_to_temporal_norm(pooled)
        temporal_feat = self.temporal_encoder(pooled.unsqueeze(2)).squeeze(2)
        return self.cnn_head(temporal_feat)
