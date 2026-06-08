"""
transformer.py
--------------
Transformer architecture cho WiFi CSI -> Human Pose Estimation.

Architecture:
    CSI tokens (B, T, N_patches, D)
        -> [Patch PE + Time PE + Node PE]   ← positional encoding
        -> Spatial Encoder  : attention qua subcarrier patches (within each time step)
        -> Temporal Encoder : attention qua time steps per-patch (NOT mean-pooled!)
        -> CLS Token        ← self-attention với toàn bộ spatio-temporal context
        -> Pose Decoder     : per-frame + cross-frame cross-attention
        -> Output: joint coordinates (B, T, N_joints, 3)
                   joint visibility logits (B, T, N_joints)  ← raw logits, NOT sigmoid

"""

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat


# ===========================================================================
# 1. Pre-LN Transformer Block
# ===========================================================================
class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer block:
        x -> LN -> MHA -> x + residual
        x -> LN -> FFN -> x + residual
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        attn_drop: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_drop,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D)
        key_padding_mask: (B, L) — True = ignore
        attn_mask: (L, L) — True = ignore (for causal)
        """
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


# ===========================================================================
# 2. Spatial Encoder
# ===========================================================================
class SpatialEncoder(nn.Module):
    """
    Mỗi time step, attend qua N_patches subcarrier patches.
    Input:  (B, T, N_patches, D)
    Output: (B, T, N_patches, D)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, N, D = x.shape
        x = rearrange(x, "b t n d -> (b t) n d")

        if key_padding_mask is not None:
            key_padding_mask = rearrange(key_padding_mask, "b t n -> (b t) n")

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        x = rearrange(x, "(b t) n d -> b t n d", b=B, t=T)
        return x


# ===========================================================================
# 3. Temporal Encoder 
# ===========================================================================
class TemporalEncoder(nn.Module):
    """
    Attend qua T time steps PER-PATCH để học temporal dynamics
    cho từng frequency band riêng biệt.

    Input:  (B, T, N_patches, D)
    Output: (B, T, N_patches, D)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T, N_patches, D)
        key_padding_mask: (B, T, N) — True = ignore (optional)
        """
        B, T, N, D = x.shape

        # Flatten B,N để attend qua time per-patch
        x = rearrange(x, "b t n d -> (b n) t d")  # (B*N, T, D)

        attn_mask = self._causal_mask(T, x.device) if self.causal else None

        if key_padding_mask is not None:
            key_padding_mask = rearrange(key_padding_mask, "b t n -> (b n) t")

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        x = self.norm(x)  # (B*N, T, D)
        x = rearrange(x, "(b n) t d -> b t n d", b=B, n=N)
        return x


# ===========================================================================
# 4. Pose Decoder — memory efficient + cross-frame temporal attention
# ===========================================================================
class PoseDecoder(nn.Module):
    """
    Decode pose từ CSI features qua cross-attention 2 stage:
      1. Per-frame cross-attention: joints attend CSI features của frame tương ứng
      2. Cross-frame temporal attention: joints attend qua time axis
         → Smooth pose prediction, catch long-range motion dependencies
         → Bidirectional cho training, causal option cho online inference

    Output: per-joint coordinates + visibility LOGITS (raw, không sigmoid)
    """

    def __init__(
        self,
        n_joints: int = 17,
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

        # Learnable joint queries
        self.joint_queries = nn.Parameter(torch.randn(n_joints, d_model))
        nn.init.trunc_normal_(self.joint_queries, std=0.02)

        # --- Stage 1: Per-frame cross-attention layers ---
        self.decoder_layers = nn.ModuleList()
        for _ in range(n_decoder_layers):
            self.decoder_layers.append(nn.ModuleDict({
                "self_attn": TransformerBlock(d_model, n_heads, ffn_mult, dropout),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm_cross": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * ffn_mult),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * ffn_mult, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        # --- Stage 2: Cross-frame temporal attention layers ---
        self.temporal_layers = nn.ModuleList()
        for _ in range(n_temporal_layers):
            self.temporal_layers.append(nn.ModuleDict({
                "self_attn": TransformerBlock(d_model, n_heads, ffn_mult, dropout),
                "temporal_attn": nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm_temporal": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * ffn_mult),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * ffn_mult, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        self.causal_temporal = causal_temporal
        self.norm_out = nn.LayerNorm(d_model)
        self.coord_head = nn.Linear(d_model, coord_dim)
        self.vis_head = nn.Linear(d_model, 1)  # raw logits

    def _temporal_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Causal mask cho cross-frame temporal attention. Shape: (T, T)."""
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        csi_features: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        csi_features: (B, T, N_patches, D)
        key_padding_mask: (B, T, N) — True = ignore

        Returns:
            joints_coord: (B, T, N_joints, coord_dim)
            joints_vis:   (B, T, N_joints) — raw logits
        """
        B, T, N, D = csi_features.shape
        J = self.n_joints

        # --- Stage 1: Per-frame cross-attention ---
        csi_flat = rearrange(csi_features, "b t n d -> b (t n) d")  # (B, T*N, D)

        # Expand joint queries: (J, D) -> (B, T, J, D) -> (B*T, J, D)
        q = repeat(self.joint_queries, "j d -> b t j d", b=B, t=T)
        q = q.reshape(B * T, J, D)

        for layer in self.decoder_layers:
            # 1a. Self-attention giữa joints (per time-step)
            q = layer["self_attn"](q)

            # 1b. Cross-attention per-frame:
            csi_per_frame = rearrange(csi_features, "b t n d -> (b t) n d")

            residual = q
            q_norm = layer["norm_cross"](q)
            q_attended, _ = layer["cross_attn"](
                query=q_norm,
                key=csi_per_frame,
                value=csi_per_frame,
            )
            q = residual + q_attended
            q = q + layer["ffn"](q)

        # q: (B*T, J, D) → (B, T, J, D)
        q = rearrange(q, "(b t) j d -> b t j d", b=B, t=T)

        # --- Stage 2: Cross-frame temporal attention ---
        # (B*J, T, D) — mỗi joint có temporal context riêng
        q = rearrange(q, "b t j d -> (b j) t d")

        temporal_mask = None
        if self.causal_temporal:
            temporal_mask = self._temporal_causal_mask(T, q.device)

        for layer in self.temporal_layers:
            # 2a. Self-attention giữa joints (per time-step)
            # Áp dụng causal mask consistency — nếu causal thì self_attn cũng mask
            q_bt = rearrange(q, "(b j) t d -> (b t) j d", b=B, j=J)
            if self.causal_temporal:
                q_bt = q_bt.detach() + (q_bt - q_bt)  # no-op để tránh warning
                # Note: self_attn giữa joints không cần causal mask vì
                # đây là cross-sectional (tất cả joints tại cùng time step)
                # Chỉ temporal attention mới cần causal
            q_bt = layer["self_attn"](q_bt)
            q = rearrange(q_bt, "(b t) j d -> (b j) t d", b=B, t=T)

            # 2b. Temporal attention: attend qua time axis (causal nếu cần)
            residual = q
            q_norm = layer["norm_temporal"](q)
            q_temp, _ = layer["temporal_attn"](
                query=q_norm,
                key=q_norm,
                value=q_norm,
                attn_mask=temporal_mask,
            )
            q = residual + q_temp
            q = q + layer["ffn"](q)

        # q: (B*J, T, D) → (B, T, J, D)
        q = rearrange(q, "(b j) t d -> b t j d", b=B, j=J)
        q = self.norm_out(q)

        coords = self.coord_head(q)              # (B, T, J, coord_dim)
        vis_logits = self.vis_head(q).squeeze(-1)  # (B, T, J) — raw logits

        return coords, vis_logits


# ===========================================================================
# 5. CLS Token Module — Active learning path
# ===========================================================================
class CLSTokenModule(nn.Module):
    """
    Learnable CLS token đi qua self-attention với temporal features.
    Cho action recognition, presence detection, identity tracking.

    multi-node: nếu N_eff = nodes * n_patches, unflatten về (B, T, nodes, N, D)
    rồi mean-pool qua nodes để giữ semantics đúng.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        n_nodes: int = 1,
        n_patches: int = 19,
        num_actions: int = 13,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_patches = n_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.cls_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.action_head = nn.Linear(d_model, num_actions)
        self.presence_head = nn.Linear(d_model, 1)

    def _unflatten_nodes(
        self, temporal_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Nếu multi-node: unflatten (B, T, nodes*N, D) → mean-pool qua nodes.
        Single-node: không thay đổi.
        """
        if self.n_nodes <= 1:
            return temporal_feat

        B, T, N_eff, D = temporal_feat.shape
        # Unflatten: (B, T, nodes*n_patches, D) -> (B, nodes, T, n_patches, D)
        # Rồi mean-pool qua nodes → (B, T, n_patches, D)
        x = rearrange(
            temporal_feat,
            "b t (nd n) d -> b nd t n d",
            nd=self.n_nodes,
            n=self.n_patches,
        )  # (B, nodes, T, n_patches, D)
        x = x.mean(dim=1)  # mean-pool qua nodes → (B, T, n_patches, D)
        return x

    def forward(
        self,
        temporal_feat: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        temporal_feat: (B, T, N_eff, D)
        key_padding_mask: (B, T*N_eff) — True = ignore
        returns: dict với cls_feat, action_logits, presence_logit
        """
        B, T, N_eff, D = temporal_feat.shape

        # Multi-node: unflatten nodes để giữ semantics
        temporal_feat = self._unflatten_nodes(temporal_feat)  # (B, T, N, D)
        _, _, N, _ = temporal_feat.shape

        # Flatten temporal features: (B, T*N, D)
        seq = rearrange(temporal_feat, "b t n d -> b (t n) d")

        # Expand cls token: (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concatenate: [CLS, seq_tokens] → (B, 1 + T*N, D)
        full_seq = torch.cat([cls_tokens, seq], dim=1)

        # Prepare mask: cls token không bị mask, seq tokens theo key_padding_mask
        full_mask = None
        if key_padding_mask is not None:
            # key_padding_mask là (B, T*N_eff) từ input
            # Sau unflatten, N_eff → N (n_patches), nên cần reshape mask tương ứng
            if self.n_nodes > 1:
                # Mask cũng cần unflatten → mean-pool logic
                # key_padding_mask: (B, T*N_eff) -> (B, T, N_eff) -> mean qua nodes
                mask_3d = key_padding_mask.reshape(B, T, N_eff)
                # Unflatten: (B, T, nodes*N) -> (B, nodes, T, N)
                mask_nodes = rearrange(
                    mask_3d, "b t (nd n) -> b nd t n",
                    nd=self.n_nodes, n=self.n_patches,
                )
                # Mean-pool qua nodes → (B, T, N) → flatten
                mask_pooled = mask_nodes.float().mean(dim=1) > 0.5
                mask_pooled = rearrange(mask_pooled, "b t n -> b (t n)")
                full_mask = torch.cat([
                    torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device),
                    mask_pooled,
                ], dim=1)
            else:
                full_mask = torch.cat([
                    torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device),
                    key_padding_mask,
                ], dim=1)

        # CLS token self-attention với toàn bộ sequence
        cls_query = full_seq[:, :1]  # (B, 1, D)
        cls_attended, _ = self.cls_attn(
            query=cls_query,
            key=full_seq,
            value=full_seq,
            key_padding_mask=full_mask,
        )

        #FFN dùng đúng pre-norm residual pattern
        # cls_attended: (B, 1, D)
        cls_feat = self.norm1(cls_attended.squeeze(1))  # (B, D)
        cls_feat = cls_feat + self.ffn(cls_feat)
        cls_feat = self.norm2(cls_feat)                 # (B, D)

        action_logits = self.action_head(cls_feat)
        presence_logit = self.presence_head(cls_feat).squeeze(-1)

        return {
            "cls_feat": cls_feat,
            "action_logits": action_logits,
            "presence_logit": presence_logit,
        }


# ===========================================================================
# 6. Full Transformer Model
# ===========================================================================
class CSITransformerPose(nn.Module):
    """
    Full model: CSI tokens -> Positional Encoding -> Spatial Encoder
                -> Temporal Encoder -> CLS Token (active) -> Pose Decoder

    Multi-node:
        KHÔNG flatten nodes vào time dimension (sai graph structure).
        Mỗi node có node embedding riêng, spatial encoder attend qua
        [nodes, patches] để học cross-node frequency correlation.
        CLS token: unflatten nodes rồi mean-pool để giữ semantics.
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
        n_joints: int = 17,
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

        # --- Positional embeddings ---
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, 1, n_patches, d_model))
        nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)

        self.time_pos_embed = nn.Parameter(torch.zeros(1, max_time, 1, d_model))
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        if n_nodes > 1:
            self.node_pos_embed = nn.Embedding(n_nodes, d_model)
            nn.init.trunc_normal_(self.node_pos_embed.weight, std=0.02)
        else:
            self.node_pos_embed = None

        # --- Encoders ---
        self.spatial_encoder = SpatialEncoder(
            d_model=d_model, n_heads=spatial_heads,
            n_layers=n_spatial_layers, ffn_mult=ffn_mult, dropout=dropout,
        )
        self.temporal_encoder = TemporalEncoder(
            d_model=d_model, n_heads=temporal_heads,
            n_layers=n_temporal_layers, ffn_mult=ffn_mult,
            dropout=dropout, causal=causal_temporal,
        )

        # --- CLS Token (active learning path) ---
        self.cls_module = CLSTokenModule(
            d_model=d_model, n_heads=spatial_heads,
            ffn_mult=ffn_mult, dropout=dropout,
            n_nodes=n_nodes, n_patches=n_patches,
            num_actions=num_actions,
        )

        # --- Pose Decoder ---
        self.pose_decoder = PoseDecoder(
            n_joints=n_joints, d_model=d_model,
            n_heads=spatial_heads,
            n_decoder_layers=n_decoder_layers,
            n_temporal_layers=n_decoder_temporal_layers,
            ffn_mult=ffn_mult, dropout=dropout,
            predict_3d=predict_3d,
            causal_temporal=causal_temporal,
        )

        # Normalization giữa spatial → temporal (ngăn distribution shift)
        self.spatial_to_temporal_norm = nn.LayerNorm(d_model)

    def add_positional_encoding(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Thêm positional encoding vào tokens.

        tokens: (B, T, N, D) — single-node
              hoặc (B, nodes, T, N, D) — multi-node

        Multi-node: cộng node embedding, merge nodes+patches cho spatial encoder.
        Single-node: cộng patch PE + time PE.
        """
        if tokens.ndim == 5:
            # Multi-node: (B, nodes, T, N, D)
            B, nodes, T, N, D = tokens.shape
            node_ids = torch.arange(nodes, device=tokens.device)
            node_pe = self.node_pos_embed(node_ids).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            tokens = tokens + node_pe  # (B, nodes, T, N, D)

            # Merge nodes + patches cho spatial encoder
            tokens = rearrange(tokens, "b nd t n d -> b t (nd n) d")
            N = N * nodes  # effective patch count
        else:
            B, T, N, D = tokens.shape

        # Patch positional encoding
        tokens = tokens + self.patch_pos_embed[:, :, :N, :]

        # Time positional encoding — explicit expand để broadcast đúng semantics
        tokens = tokens + self.time_pos_embed[:, :T].expand(-1, -1, N, -1)

        return tokens

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        tokens: (B, T, N_patches, d_model) — single-node
              hoặc (B, nodes, T, N_patches, d_model) — multi-node

        key_padding_mask: (B, T, N) hoặc (B, nodes, T, N) — True = ignore

        returns: dict với keys:
            'coords':        (B, T, N_joints, coord_dim)
            'vis_logits':    (B, T, N_joints) — raw logits
            'action_logits': (B, num_actions)
            'presence_logit': (B,)
            'cls_feat':      (B, d_model)
            'spatial_feat':  (B, T, N_eff, D)
            'temporal_feat': (B, T, N_eff, D)
        """
        # Add positional encoding
        tokens = self.add_positional_encoding(tokens)

        B, T, N_eff, D = tokens.shape

        # key_padding_mask reshape nếu cần
        if key_padding_mask is not None:
            if key_padding_mask.ndim == 4:
                key_padding_mask = rearrange(key_padding_mask, "b nd t n -> b t (nd n)")
            # (B, T, N_eff)

        # --- Spatial encoding (frequency correlation) ---
        spatial_feat = self.spatial_encoder(
            tokens, key_padding_mask=key_padding_mask
        )  # (B, T, N_eff, D)

        # Normalization giữa spatial → temporal
        spatial_feat = self.spatial_to_temporal_norm(spatial_feat)

        # --- Temporal encoding (motion dynamics) — per-patch ---
        temporal_feat = self.temporal_encoder(
            spatial_feat, key_padding_mask=key_padding_mask
        )  # (B, T, N_eff, D)

        # --- CLS Token (active learning path) ---
        cls_mask = None
        if key_padding_mask is not None:
            cls_mask = rearrange(key_padding_mask, "b t n -> b (t n)")

        cls_out = self.cls_module(temporal_feat, key_padding_mask=cls_mask)

        # --- Pose decoding ---
        # Unflatten nodes về single-node dimension cho decoder
        decoder_feat = temporal_feat  # (B, T, N_eff, D)
        if self.n_nodes > 1:
            # temporal_feat: (B, T, nodes*n_patches, D) → CLS đã handle unflatten
            # Decoder cần: unflatten nodes, mean-pool, hoặc giữ nguyên
            # Giữ nguyên (nodes+n_patches) — decoder attend qua tất cả
            pass
        coords, vis_logits = self.pose_decoder(
            decoder_feat, key_padding_mask=key_padding_mask
        )

        return {
            "coords":         coords,
            "vis_logits":     vis_logits,
            "action_logits":  cls_out["action_logits"],
            "presence_logit": cls_out["presence_logit"],
            "cls_feat":       cls_out["cls_feat"],
            "spatial_feat":   spatial_feat,
            "temporal_feat":  temporal_feat,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Quick test
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, N_patches, D = 4, 100, 19, 256

    model = CSITransformerPose(
        n_patches=N_patches,
        d_model=D,
        spatial_heads=8,
        temporal_heads=8,
        n_spatial_layers=4,
        n_temporal_layers=4,
        n_decoder_layers=3,
        n_decoder_temporal_layers=2,
        n_joints=17,
        predict_3d=True,
        n_nodes=1,
    )

    tokens = torch.randn(B, T, N_patches, D)
    out = model(tokens)

    print(f"Params: {model.count_params():,}")
    print(f"Input tokens:  {tuple(tokens.shape)}")
    print(f"coords:        {tuple(out['coords'].shape)}")
    print(f"vis_logits:    {tuple(out['vis_logits'].shape)}")
    print(f"action_logits: {tuple(out['action_logits'].shape)}")
    print(f"presence_logit:{tuple(out['presence_logit'].shape)}")
    print(f"cls_feat:      {tuple(out['cls_feat'].shape)}")

    # Test with key_padding_mask
    mask = torch.zeros(B, T, N_patches, dtype=torch.bool)
    mask[:, 90:, :] = True
    out_masked = model(tokens, key_padding_mask=mask)
    print(f"\nWith mask: coords {tuple(out_masked['coords'].shape)}")

    # Test multi-node
    model_multi = CSITransformerPose(n_nodes=4, n_patches=N_patches, d_model=D, n_joints=17)
    tokens_multi = torch.randn(B, 4, T, N_patches, D)
    out_multi = model_multi(tokens_multi)
    print(f"\nMulti-node: coords {tuple(out_multi['coords'].shape)}")
    print(f"Multi-node: cls_feat {tuple(out_multi['cls_feat'].shape)}")
