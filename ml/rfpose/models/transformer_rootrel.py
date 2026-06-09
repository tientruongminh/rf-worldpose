"""
transformer_rootrel.py
----------------------
Root-Relative Coordinate variant of CSITransformerPose.

Key idea: decompose absolute pose prediction into two subtasks:
    1. Root position (pelvis center) — predicted from pooled joint features
    2. Joint offsets relative to root — predicted per-joint

coords_abs = root.unsqueeze(2) + offsets

Benefits:
    - Simplifies learning: offset magnitudes are small and bounded
    - Root position captures global translation (hardest part of WiFi sensing)
    - Offsets capture body articulation (easier, more structured)
    - Empirically shown to reduce MPJPE in vision-based HPE literature

Imports base classes from transformer.py — no duplication.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from rfpose.models.transformer import (
    TransformerBlock,
    SpatialEncoder,
    TemporalEncoder,
    CLSTokenModule,
)


class PoseDecoderRootRel(nn.Module):
    """
    Pose decoder with root-relative coordinate prediction.

    Output decomposition:
        root:    (B, T, 3) — absolute pelvis position
        offsets: (B, T, J, 3) — per-joint displacement from root
        coords:  (B, T, J, 3) — reconstructed absolute = root + offsets
    """

    def __init__(
        self,
        n_joints: int = 13,
        d_model: int = 256,
        n_heads: int = 8,
        n_decoder_layers: int = 3,
        n_temporal_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        predict_3d: bool = True,
        causal_temporal: bool = False,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.predict_3d = predict_3d
        coord_dim = 3 if predict_3d else 2

        self.joint_queries = nn.Parameter(torch.randn(n_joints, d_model))
        nn.init.trunc_normal_(self.joint_queries, std=0.02)

        # Stage 1: Per-frame cross-attention layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(n_decoder_layers):
            self.decoder_layers.append(nn.ModuleDict({
                "self_attn": TransformerBlock(d_model, n_heads, ffn_mult, dropout),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=n_heads,
                    dropout=dropout, batch_first=True,
                ),
                "norm_cross": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * ffn_mult),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model * ffn_mult, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        # Stage 2: Cross-frame temporal attention
        self.temporal_layers = nn.ModuleList()
        for _ in range(n_temporal_layers):
            self.temporal_layers.append(nn.ModuleDict({
                "self_attn": TransformerBlock(d_model, n_heads, ffn_mult, dropout),
                "temporal_attn": nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=n_heads,
                    dropout=dropout, batch_first=True,
                ),
                "norm_temporal": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * ffn_mult),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model * ffn_mult, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        self.causal_temporal = causal_temporal
        self.norm_out = nn.LayerNorm(d_model)

        # Separate heads for root and offsets
        self.offset_head = nn.Linear(d_model, coord_dim)
        self.root_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, coord_dim),
        )
        self.vis_head = nn.Linear(d_model, 1)

    def _temporal_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        csi_features: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T, N, D = csi_features.shape
        J = self.n_joints

        # Stage 1: Per-frame cross-attention
        q = repeat(self.joint_queries, "j d -> b t j d", b=B, t=T)
        q = q.reshape(B * T, J, D)

        for layer in self.decoder_layers:
            q = layer["self_attn"](q)

            csi_per_frame = rearrange(csi_features, "b t n d -> (b t) n d")
            residual = q
            q_norm = layer["norm_cross"](q)
            q_attended, _ = layer["cross_attn"](
                query=q_norm, key=csi_per_frame, value=csi_per_frame,
            )
            q = residual + q_attended
            q = q + layer["ffn"](q)

        q = rearrange(q, "(b t) j d -> b t j d", b=B, t=T)

        # Stage 2: Cross-frame temporal attention
        q = rearrange(q, "b t j d -> (b j) t d")

        temporal_mask = None
        if self.causal_temporal:
            temporal_mask = self._temporal_causal_mask(T, q.device)

        for layer in self.temporal_layers:
            q_bt = rearrange(q, "(b j) t d -> (b t) j d", b=B, j=J)
            q_bt = layer["self_attn"](q_bt)
            q = rearrange(q_bt, "(b t) j d -> (b j) t d", b=B, t=T)

            residual = q
            q_norm = layer["norm_temporal"](q)
            q_temp, _ = layer["temporal_attn"](
                query=q_norm, key=q_norm, value=q_norm,
                attn_mask=temporal_mask,
            )
            q = residual + q_temp
            q = q + layer["ffn"](q)

        q = rearrange(q, "(b j) t d -> b t j d", b=B, j=J)
        q = self.norm_out(q)

        # Root-relative decomposition
        offsets = self.offset_head(q)          # (B, T, J, coord_dim)
        root_feat = q.mean(dim=2)              # (B, T, D) — pool over joints
        root = self.root_head(root_feat)       # (B, T, coord_dim)
        coords = root.unsqueeze(2) + offsets   # (B, T, J, coord_dim)
        vis_logits = self.vis_head(q).squeeze(-1)

        return {
            "coords": coords,
            "vis_logits": vis_logits,
            "root": root,
            "offsets": offsets,
        }


class CSITransformerPoseRootRel(nn.Module):
    """
    Full Transformer model with root-relative pose decoder.

    Same encoder architecture as CSITransformerPose, but PoseDecoder
    is replaced with PoseDecoderRootRel.

    Output dict includes 'root' and 'offsets' in addition to standard keys.
    """

    def __init__(
        self,
        n_patches: int = 19,
        d_model: int = 256,
        spatial_heads: int = 8,
        temporal_heads: int = 8,
        n_spatial_layers: int = 4,
        n_temporal_layers: int = 4,
        n_decoder_layers: int = 3,
        n_decoder_temporal_layers: int = 2,
        n_joints: int = 13,
        predict_3d: bool = True,
        causal_temporal: bool = False,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        max_time: int = 500,
        n_nodes: int = 1,
        num_actions: int = 13,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_joints = n_joints
        self.n_nodes = n_nodes
        self.n_patches = n_patches

        # Positional embeddings (same as base)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, 1, n_patches, d_model))
        nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)

        self.time_pos_embed = nn.Parameter(torch.zeros(1, max_time, 1, d_model))
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        if n_nodes > 1:
            self.node_pos_embed = nn.Embedding(n_nodes, d_model)
            nn.init.trunc_normal_(self.node_pos_embed.weight, std=0.02)
        else:
            self.node_pos_embed = None

        # Encoders (same as base)
        self.spatial_encoder = SpatialEncoder(
            d_model=d_model, n_heads=spatial_heads,
            n_layers=n_spatial_layers, ffn_mult=ffn_mult, dropout=dropout,
        )
        self.temporal_encoder = TemporalEncoder(
            d_model=d_model, n_heads=temporal_heads,
            n_layers=n_temporal_layers, ffn_mult=ffn_mult,
            dropout=dropout, causal=causal_temporal,
        )

        # CLS Token (same as base)
        self.cls_module = CLSTokenModule(
            d_model=d_model, n_heads=spatial_heads,
            ffn_mult=ffn_mult, dropout=dropout,
            n_nodes=n_nodes, n_patches=n_patches,
            num_actions=num_actions,
        )

        # Root-relative Pose Decoder
        self.pose_decoder = PoseDecoderRootRel(
            n_joints=n_joints, d_model=d_model,
            n_heads=spatial_heads,
            n_decoder_layers=n_decoder_layers,
            n_temporal_layers=n_decoder_temporal_layers,
            ffn_mult=ffn_mult, dropout=dropout,
            predict_3d=predict_3d,
            causal_temporal=causal_temporal,
        )

        self.spatial_to_temporal_norm = nn.LayerNorm(d_model)

    def add_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 5:
            B, nodes, T, N, D = tokens.shape
            node_ids = torch.arange(nodes, device=tokens.device)
            node_pe = self.node_pos_embed(node_ids).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            tokens = tokens + node_pe
            tokens = rearrange(tokens, "b nd t n d -> b t (nd n) d")
            N = N * nodes
        else:
            B, T, N, D = tokens.shape

        tokens = tokens + self.patch_pos_embed[:, :, :N, :]
        tokens = tokens + self.time_pos_embed[:, :T].expand(-1, -1, N, -1)
        return tokens

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self.add_positional_encoding(tokens)
        B, T, N_eff, D = tokens.shape

        if key_padding_mask is not None:
            if key_padding_mask.ndim == 4:
                key_padding_mask = rearrange(key_padding_mask, "b nd t n -> b t (nd n)")

        spatial_feat = self.spatial_encoder(tokens, key_padding_mask=key_padding_mask)
        spatial_feat = self.spatial_to_temporal_norm(spatial_feat)
        temporal_feat = self.temporal_encoder(spatial_feat, key_padding_mask=key_padding_mask)

        cls_mask = None
        if key_padding_mask is not None:
            cls_mask = rearrange(key_padding_mask, "b t n -> b (t n)")
        cls_out = self.cls_module(temporal_feat, key_padding_mask=cls_mask)

        decoder_out = self.pose_decoder(temporal_feat, key_padding_mask=key_padding_mask)

        return {
            "coords":         decoder_out["coords"],
            "vis_logits":     decoder_out["vis_logits"],
            "root":           decoder_out["root"],
            "offsets":        decoder_out["offsets"],
            "action_logits":  cls_out["action_logits"],
            "presence_logit": cls_out["presence_logit"],
            "cls_feat":       cls_out["cls_feat"],
            "spatial_feat":   spatial_feat,
            "temporal_feat":  temporal_feat,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
