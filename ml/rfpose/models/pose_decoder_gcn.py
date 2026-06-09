"""
pose_decoder_gcn.py
-------------------
GCN-based pose decoder inspired by DT-Pose (Chen et al., 2025).

Key ideas from DT-Pose adapted here:
    1. Learnable task prompts (one per joint) — similar to joint queries
    2. Graph Convolutional Network using skeleton adjacency matrix
    3. Transformer refinement layers after GCN
    4. Root-relative prediction (from our improvement #5)

Architecture:
    Encoder features [B, T, d] (from CSITokenizer + Transformer encoder)
      → expand per joint via task prompts → [B, T, J, d]
      → 3-layer GCN (skeleton topology) → [B, T, J, d]
      → 3-layer Transformer (inter-joint attention) → [B, T, J, d]
      → root head → [B, T, 3]
      → offset head → [B, T, J, 3]
      → coords = root + offsets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


SKELETON_13 = [
    [0, 1], [0, 4],           # head → shoulders
    [1, 2], [2, 3],           # left arm
    [4, 5], [5, 6],           # right arm
    [1, 7], [7, 8], [8, 9],   # left leg
    [4, 10], [10, 11], [11, 12],  # right leg
    [1, 4],                    # shoulder bridge
    [7, 10],                   # hip bridge
]


def build_adjacency(n_joints: int, connections: list[list[int]], self_loops: bool = True) -> torch.Tensor:
    adj = torch.zeros(n_joints, n_joints)
    for i, j in connections:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    if self_loops:
        adj += torch.eye(n_joints)
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / deg


class GraphConvLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: [*, J, d], adj: [J, J] (normalized)."""
        residual = x
        x = torch.matmul(adj, x)
        x = self.fc(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class JointTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*T, J, d]"""
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + x2)
        x = self.norm2(x + self.ffn(x))
        return x


class GCNPoseDecoder(nn.Module):
    """GCN skeleton decoder with root-relative prediction."""

    def __init__(
        self,
        n_joints: int = 13,
        d_model: int = 256,
        n_gcn_layers: int = 3,
        n_tf_layers: int = 3,
        n_heads: int = 4,
        coord_dim: int = 3,
        dropout: float = 0.1,
        num_actions: int = 28,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.d_model = d_model

        self.pose_prompt = nn.Parameter(torch.zeros(d_model, n_joints))
        trunc_normal_(self.pose_prompt, std=0.02)

        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(d_model, dropout) for _ in range(n_gcn_layers)
        ])

        self.tf_layers = nn.ModuleList([
            JointTransformerLayer(d_model, n_heads, dropout) for _ in range(n_tf_layers)
        ])

        self.offset_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, coord_dim),
        )

        self.root_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, coord_dim),
        )

        self.vis_head = nn.Linear(d_model, 1)

        self.action_attn = nn.Linear(d_model, 1)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_actions),
        )

        adj = build_adjacency(n_joints, SKELETON_13)
        self.register_buffer("adj", adj)

    def forward(self, encoder_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        encoder_features: [B, T, d_model] — pooled from spatial+temporal encoder.
        Returns dict with coords, vis_logits, action_logits, root, offsets.
        """
        B, T, d = encoder_features.shape

        x = encoder_features.unsqueeze(2).expand(B, T, self.n_joints, d)
        prompt = self.pose_prompt.unsqueeze(0).unsqueeze(0).expand(B, T, d, self.n_joints)
        x = x + prompt.permute(0, 1, 3, 2)

        BT = B * T
        x = x.reshape(BT, self.n_joints, d)

        for gcn in self.gcn_layers:
            x = gcn(x, self.adj)

        for tf in self.tf_layers:
            x = tf(x)

        x = x.reshape(B, T, self.n_joints, d)

        offsets = self.offset_head(x)
        root_feat = x.mean(dim=2)
        root = self.root_head(root_feat)
        coords = root.unsqueeze(2) + offsets
        vis_logits = self.vis_head(x).squeeze(-1)

        w = torch.softmax(self.action_attn(root_feat), dim=1)
        pooled = (root_feat * w).sum(dim=1)
        action_logits = self.action_head(pooled)

        return {
            "coords": coords,
            "vis_logits": vis_logits,
            "action_logits": action_logits,
            "root": root,
            "offsets": offsets,
        }


class CSITransformerPoseGCN(nn.Module):
    """Full model: CSI encoder (from CSITransformerPose) + GCN pose decoder.

    Reuses spatial_encoder and temporal_encoder from the base model,
    replaces the cross-attention PoseDecoder with GCN-based decoder.
    """

    def __init__(
        self,
        n_patches: int,
        d_model: int = 256,
        spatial_heads: int = 8,
        temporal_heads: int = 8,
        n_spatial_layers: int = 4,
        n_temporal_layers: int = 4,
        n_gcn_layers: int = 3,
        n_gcn_tf_layers: int = 3,
        n_joints: int = 13,
        predict_3d: bool = True,
        causal_temporal: bool = False,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        n_nodes: int = 1,
        num_actions: int = 28,
    ):
        super().__init__()
        from rfpose.models.transformer import SpatialEncoder, TemporalEncoder

        self.spatial_encoder = SpatialEncoder(
            d_model=d_model,
            n_heads=spatial_heads,
            n_layers=n_spatial_layers,
            dropout=dropout,
            ffn_mult=ffn_mult,
        )

        self.spatial_to_temporal_norm = nn.LayerNorm(d_model)

        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            n_heads=temporal_heads,
            n_layers=n_temporal_layers,
            causal=causal_temporal,
            dropout=dropout,
            ffn_mult=ffn_mult,
        )

        self.gcn_decoder = GCNPoseDecoder(
            n_joints=n_joints,
            d_model=d_model,
            n_gcn_layers=n_gcn_layers,
            n_tf_layers=n_gcn_tf_layers,
            n_heads=4,
            coord_dim=3 if predict_3d else 2,
            dropout=dropout,
            num_actions=num_actions,
        )

    def forward(self, csi_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """csi_tokens: [B, T, N_patches, d_model] from CSITokenizer."""
        B, T, N, d = csi_tokens.shape

        # SpatialEncoder: [B, T, N, D] → [B, T, N, D]
        spatial_out = self.spatial_encoder(csi_tokens)
        spatial_out = self.spatial_to_temporal_norm(spatial_out)

        # TemporalEncoder: [B, T, N, D] → [B, T, N, D]
        temporal_out = self.temporal_encoder(spatial_out)

        # Pool patches → [B, T, D] for GCN decoder
        temporal_pooled = temporal_out.mean(dim=2)

        return self.gcn_decoder(temporal_pooled)
