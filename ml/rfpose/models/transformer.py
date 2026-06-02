"""
transformer.py
--------------
Transformer architecture cho WiFi CSI -> Human Pose Estimation.

Architecture:
    CSI tokens (B, T, N_patches, D)
        -> Spatial Encoder  : attention qua subcarrier patches (within each time step)
        -> Temporal Encoder : attention qua time steps (sequence modeling)
        -> Pose Decoder     : cross-attention với learnable joint queries
        -> Output: joint coordinates (B, T_out, N_joints, 3)  # x, y, z
                   joint visibility  (B, T_out, N_joints)

Lý do dùng kiến trúc hai-stage Spatial+Temporal:
    - CSI có 2 chiều thông tin: subcarrier (spatial/frequency) và time
    - Spatial encoder học correlation giữa các frequency band
    - Temporal encoder học motion dynamics theo thời gian
    - Tách 2 stage giảm complexity từ O((T*N)^2) xuống O(T^2 + N^2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# 1. Pre-LN Transformer Block (ổn định hơn Post-LN cho training sâu)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer block:
        x -> LN -> MHA -> x + residual
        x -> LN -> FFN -> x + residual

    Args:
        d_model:    embedding dimension
        n_heads:    số attention heads
        ffn_mult:   FFN hidden dim = d_model * ffn_mult
        dropout:    dropout rate
        attn_drop:  dropout trong attention weights
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
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

        self.drop_path = nn.Identity()  # có thể thay bằng StochasticDepth nếu cần

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D) — sequence of L tokens
        """
        # Multi-head self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = residual + self.drop_path(x)

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.drop_path(x)

        return x


# ---------------------------------------------------------------------------
# 2. Spatial Encoder — attend qua N_patches (subcarrier axis)
# ---------------------------------------------------------------------------
class SpatialEncoder(nn.Module):
    """
    Mỗi time step, attend qua N_patches subcarrier patches.
    Học được: frequency correlation, multipath patterns, antenna diversity.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_patches, D)
        """
        B, T, N, D = x.shape

        # Flatten B,T để attention chạy trên N_patches
        x = rearrange(x, "b t n d -> (b t) n d")

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = rearrange(x, "(b t) n d -> b t n d", b=B, t=T)
        return x


# ---------------------------------------------------------------------------
# 3. Temporal Encoder — attend qua T (time axis)
# ---------------------------------------------------------------------------
class TemporalEncoder(nn.Module):
    """
    Attend qua T time steps để học temporal dynamics (motion, gait, gesture).
    Dùng causal mask (optional) cho online inference.

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
        causal: bool = False,  # True nếu muốn online inference
    ):
        super().__init__()
        self.causal = causal
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper triangular mask để prevent attending to future."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask  # True = masked (ignored)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_patches, D)
        """
        B, T, N, D = x.shape

        # Mean pool qua subcarrier -> (B, T, D) để attend qua time
        # (Giữ thông tin subcarrier trong spatial encoder, temporal chỉ cần summary)
        x_temp = x.mean(dim=2)  # (B, T, D)

        attn_mask = self._causal_mask(T, x.device) if self.causal else None

        for layer in self.layers:
            x_temp = layer(x_temp, attn_mask=attn_mask)

        x_temp = self.norm(x_temp)  # (B, T, D)

        # Broadcast lại: cộng temporal context vào mỗi subcarrier patch
        x = x + x_temp.unsqueeze(2)  # (B, T, N, D)
        return x


# ---------------------------------------------------------------------------
# 4. Pose Decoder — cross-attention: joint queries <-> CSI features
# ---------------------------------------------------------------------------
class PoseDecoder(nn.Module):
    """
    Decode pose từ CSI features qua cross-attention.

    Mỗi joint có một learnable query vector.
    Cross-attention: query = joint queries, key/value = CSI tokens.

    Tư tưởng: model phải "look up" thông tin liên quan đến từng joint
    từ tập hợp CSI features => interpretable attention maps.

    Output: per-joint coordinates + visibility score

    Args:
        n_joints:       số keypoint (17 = COCO, 33 = MediaPipe)
        d_model:        embedding dimension
        n_heads:        attention heads
        n_decoder_layers: số cross-attention layers
        predict_3d:     True -> (x,y,z), False -> (x,y) only
    """

    def __init__(
        self,
        n_joints: int = 17,
        d_model: int = 256,
        n_heads: int = 8,
        n_decoder_layers: int = 3,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        predict_3d: bool = True,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.predict_3d = predict_3d
        coord_dim = 3 if predict_3d else 2

        # Learnable joint queries — mỗi joint là một learned prototype
        self.joint_queries = nn.Parameter(torch.randn(n_joints, d_model))
        nn.init.trunc_normal_(self.joint_queries, std=0.02)

        # Cross-attention layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(n_decoder_layers):
            self.decoder_layers.append(nn.ModuleDict({
                # Self-attention giữa các joints (học joint dependencies: hip-knee-ankle)
                "self_attn": TransformerBlock(d_model, n_heads, ffn_mult, dropout),
                # Cross-attention: joints attend CSI features
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

        self.norm_out = nn.LayerNorm(d_model)

        # Output heads
        self.coord_head = nn.Linear(d_model, coord_dim)   # (x, y) hoặc (x, y, z)
        self.vis_head   = nn.Linear(d_model, 1)           # visibility score

    def forward(
        self,
        csi_features: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        csi_features: (B, T, N_patches, D)
        returns:
            joints_coord: (B, T, N_joints, coord_dim)
            joints_vis:   (B, T, N_joints)

        Per-frame decoding: mỗi time step t có joint queries attend vào
        toàn bộ spatio-temporal context (B, T*N, D).
        Điều này giữ nguyên temporal gradient flow từ TemporalEncoder,
        tránh bug repeat-pose khiến temporal smoothness loss = 0.
        """
        B, T, N, D = csi_features.shape

        # Flatten T,N -> (B, T*N, D) để joints attend toàn bộ spatio-temporal context
        csi_flat = rearrange(csi_features, "b t n d -> b (t n) d")  # (B, T*N, D)

        # Expand joint queries: (N_joints, D) -> (B*T, N_joints, D)
        # Mỗi time step có bộ queries riêng (nhưng share weights)
        q = repeat(self.joint_queries, "j d -> (b t) j d", b=B, t=T)

        # Expand csi_flat để match (B*T, T*N, D)
        csi_flat_bt = repeat(csi_flat, "b l d -> (b t) l d", t=T)

        for layer in self.decoder_layers:
            # 1. Self-attention giữa joints (trong từng time step)
            q = layer["self_attn"](q)

            # 2. Cross-attention: joints query toàn bộ CSI context
            residual = q
            q_norm = layer["norm_cross"](q)
            q_attended, _ = layer["cross_attn"](
                query=q_norm,
                key=csi_flat_bt,
                value=csi_flat_bt,
            )
            q = residual + q_attended

            # 3. FFN
            q = q + layer["ffn"](q)

        q = self.norm_out(q)  # (B*T, N_joints, D)

        # Predict per-frame coordinates và visibility
        coords = self.coord_head(q)          # (B*T, N_joints, coord_dim)
        vis    = self.vis_head(q).squeeze(-1) # (B*T, N_joints)
        vis    = torch.sigmoid(vis)

        # Reshape về (B, T, N_joints, ...)
        coords = rearrange(coords, "(b t) j c -> b t j c", b=B, t=T)
        vis    = rearrange(vis,    "(b t) j -> b t j",     b=B, t=T)

        return coords, vis


# ---------------------------------------------------------------------------
# 5. Full Transformer Model
# ---------------------------------------------------------------------------
class CSITransformerPose(nn.Module):
    """
    Full model: CSI tokens -> Spatial Encoder -> Temporal Encoder -> Pose Decoder

    Args:
        n_patches:          từ CSITokenizer.n_patches (default 19 cho 114 sub, patch=6)
        d_model:            embedding dimension
        spatial_heads:      attention heads trong spatial encoder
        temporal_heads:     attention heads trong temporal encoder
        n_spatial_layers:   số spatial encoder layers
        n_temporal_layers:  số temporal encoder layers
        n_decoder_layers:   số pose decoder layers
        n_joints:           17 (COCO) hoặc 33 (MediaPipe)
        predict_3d:         predict 3D coordinates hay 2D
        causal:             causal temporal attention cho online inference
        dropout:            dropout rate
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
        n_joints: int = 17,
        predict_3d: bool = True,
        causal: bool = False,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_joints = n_joints
        coord_dim = 3 if predict_3d else 2

        self.spatial_encoder = SpatialEncoder(
            d_model=d_model,
            n_heads=spatial_heads,
            n_layers=n_spatial_layers,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            n_heads=temporal_heads,
            n_layers=n_temporal_layers,
            ffn_mult=ffn_mult,
            dropout=dropout,
            causal=causal,
        )

        self.pose_decoder = PoseDecoder(
            n_joints=n_joints,
            d_model=d_model,
            n_heads=spatial_heads,
            n_decoder_layers=n_decoder_layers,
            ffn_mult=ffn_mult,
            dropout=dropout,
            predict_3d=predict_3d,
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        tokens: (B, T, N_patches, d_model) — output của CSITokenizer
        returns: dict với keys:
            'coords':  (B, T, N_joints, coord_dim)  — predicted coordinates
            'vis':     (B, T, N_joints)              — visibility score [0,1]
            'spatial_feat':  (B, T, N_patches, D)   — intermediate feature (dùng cho distillation)
            'temporal_feat': (B, T, N_patches, D)
        """
        # Spatial encoding (frequency correlation)
        spatial_feat = self.spatial_encoder(tokens)      # (B, T, N, D)

        # Temporal encoding (motion dynamics)
        temporal_feat = self.temporal_encoder(spatial_feat)  # (B, T, N, D)

        # Pose decoding
        coords, vis = self.pose_decoder(temporal_feat)   # (B, T, J, C), (B, T, J)

        return {
            "coords":        coords,
            "vis":           vis,
            "spatial_feat":  spatial_feat,
            "temporal_feat": temporal_feat,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
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
        n_joints=17,
        predict_3d=True,
    )

    tokens = torch.randn(B, T, N_patches, D)
    out = model(tokens)

    print(f"Params: {model.count_params():,}")
    print(f"Input tokens:  {tuple(tokens.shape)}")
    print(f"coords:        {tuple(out['coords'].shape)}")   # (4, 100, 17, 3)
    print(f"vis:           {tuple(out['vis'].shape)}")      # (4, 100, 17)