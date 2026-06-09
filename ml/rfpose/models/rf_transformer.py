import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from einops import rearrange, repeat


class RunningNorm(nn.Module):
    def __init__(self, n_subcarriers: int = 270, n_channels: int = 2):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(n_subcarriers, n_channels))
        self.register_buffer("running_var",  torch.ones(n_subcarriers, n_channels))
        self.register_buffer("initialized",  torch.tensor(False))

    def forward(self, x):
        if self.initialized:
            mean = self.running_mean.T.reshape(1, 2, 1, 270)
            var  = self.running_var.T.reshape(1, 2, 1, 270)
            x = (x - mean) / (var.sqrt() + 1e-6)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 6, d_model: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.register_buffer("pos_embed", torch.zeros(1, 1, 45, d_model))
        self.proj = nn.Sequential(
            nn.Linear(patch_size * 2, d_model, bias=True),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        N, C, T, S = x.shape
        n_patches = S // self.patch_size
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(N, T, n_patches, self.patch_size * 2)
        x = self.proj(x) + self.pos_embed
        return x


class CSITokenizer(nn.Module):
    def __init__(self, n_subcarriers: int = 270, patch_size: int = 6, d_model: int = 256):
        super().__init__()
        self.norm        = RunningNorm(n_subcarriers)
        self.patch_embed = PatchEmbed(patch_size, d_model)
        self.register_buffer("temporal_pe", torch.zeros(1, 70, 1, d_model))

    def forward(self, x):
        x = self.norm(x)
        x = self.patch_embed(x)
        T = x.shape[1]
        return x + self.temporal_pe[:, :T, :, :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# SpatialEncoder — per-timestep patch attention
# ──────────────────────────────────────────────

class SpatialEncoder(nn.Module):
    """
    Attend qua N_patches trong mỗi time step.
    (B, T, N, D) → (B, T, N, D)
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=4, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, N, D = x.shape
        x = rearrange(x, "b t n d -> (b t) n d")
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return rearrange(x, "(b t) n d -> b t n d", b=B, t=T)


# ──────────────────────────────────────────────
# TemporalEncoder — per-patch temporal attention
# ──────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Attend qua T time steps PER-PATCH.
    → KHÔNG mean pool → giữ full (B, T, N, D)
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=4, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, N, D = x.shape
        x = rearrange(x, "b t n d -> (b n) t d")   # per-patch
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return rearrange(x, "(b n) t d -> b t n d", b=B, n=N)


# ──────────────────────────────────────────────
# PoseDecoder — cross-attention + temporal smoothing
# ──────────────────────────────────────────────

class PoseDecoder(nn.Module):
    """
    2-stage decoder từ transformer.py:
        Stage 1: joints cross-attend CSI features per-frame
        Stage 2: joints attend qua time axis (temporal smoothing)

    Outputs:
        coords:     (B, T, J, 3)
        vis_logits: (B, T, J)   — raw logits (không sigmoid)
    """
    def __init__(self, n_joints=13, d_model=256, n_heads=8,
                 n_decoder_layers=3, n_temporal_layers=2,
                 ffn_mult=4, dropout=0.1):
        super().__init__()
        self.n_joints = n_joints

        # Learnable joint queries — mỗi joint có "identity" riêng
        self.joint_queries = nn.Parameter(torch.randn(n_joints, d_model))
        nn.init.trunc_normal_(self.joint_queries, std=0.02)

        # Stage 1: per-frame cross-attention layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(n_decoder_layers):
            self.decoder_layers.append(nn.ModuleDict({
                "self_attn":   TransformerBlock(d_model, n_heads, ffn_mult, dropout),
                "cross_attn":  nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "norm_cross":  nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * ffn_mult),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model * ffn_mult, d_model), nn.Dropout(dropout),
                ),
            }))

        # Stage 2: cross-frame temporal attention
        self.temporal_layers = nn.ModuleList()
        for _ in range(n_temporal_layers):
            self.temporal_layers.append(nn.ModuleDict({
                "self_attn":      TransformerBlock(d_model, n_heads, ffn_mult, dropout),
                "temporal_attn":  nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "norm_temporal":  nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * ffn_mult),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model * ffn_mult, d_model), nn.Dropout(dropout),
                ),
            }))

        self.norm_out    = nn.LayerNorm(d_model)
        self.coord_head  = nn.Linear(d_model, 3)
        self.vis_head    = nn.Linear(d_model, 1)   # raw logits

    def forward(self, csi_features):
        """
        csi_features: (B, T, N, D)
        returns:
            coords:     (B, T, J, 3)
            vis_logits: (B, T, J)
        """
        B, T, N, D = csi_features.shape
        J = self.n_joints

        # Expand learnable joint queries
        q = repeat(self.joint_queries, "j d -> b t j d", b=B, t=T)
        q = rearrange(q, "b t j d -> (b t) j d")

        # Stage 1: per-frame cross-attention
        csi_per_frame = rearrange(csi_features, "b t n d -> (b t) n d")
        for layer in self.decoder_layers:
            q = layer["self_attn"](q)
            residual = q
            q_attended, _ = layer["cross_attn"](
                query=layer["norm_cross"](q),
                key=csi_per_frame, value=csi_per_frame,
            )
            q = residual + q_attended
            q = q + layer["ffn"](q)

        q = rearrange(q, "(b t) j d -> b t j d", b=B, t=T)

        # Stage 2: cross-frame temporal attention
        q = rearrange(q, "b t j d -> (b j) t d")
        for layer in self.temporal_layers:
            q_bt = rearrange(q, "(b j) t d -> (b t) j d", b=B, j=J)
            q_bt = layer["self_attn"](q_bt)
            q = rearrange(q_bt, "(b t) j d -> (b j) t d", b=B, t=T)

            residual = q
            q_norm = layer["norm_temporal"](q)
            q_temp, _ = layer["temporal_attn"](q_norm, q_norm, q_norm)
            q = residual + q_temp
            q = q + layer["ffn"](q)

        q = rearrange(q, "(b j) t d -> b t j d", b=B, j=J)
        q = self.norm_out(q)

        coords     = self.coord_head(q)              # (B, T, J, 3)
        vis_logits = self.vis_head(q).squeeze(-1)    # (B, T, J)
        return coords, vis_logits


# ──────────────────────────────────────────────
# CLSTokenModule — thay ActionHead Linear
# ──────────────────────────────────────────────

class CLSTokenModule(nn.Module):
    """
    CLS token attend qua toàn bộ spatio-temporal sequence.
    → action classification + presence detection.
    """
    def __init__(self, d_model=256, n_heads=8, ffn_mult=4,
                 dropout=0.1, num_actions=28):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.cls_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1    = nn.LayerNorm(d_model)
        self.ffn      = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model), nn.Dropout(dropout),
        )
        self.norm2          = nn.LayerNorm(d_model)
        self.action_head    = nn.Linear(d_model, num_actions)
        self.presence_head  = nn.Linear(d_model, 1)

    def forward(self, temporal_feat):
        """
        temporal_feat: (B, T, N, D)
        returns: action_logits (B, num_actions), presence_logit (B,)
        """
        B, T, N, D = temporal_feat.shape
        seq = rearrange(temporal_feat, "b t n d -> b (t n) d")

        cls = self.cls_token.expand(B, -1, -1)
        full_seq = torch.cat([cls, seq], dim=1)   # (B, 1+T*N, D)

        cls_out, _ = self.cls_attn(
            query=full_seq[:, :1],
            key=full_seq, value=full_seq,
        )
        cls_feat = self.norm1(cls_out.squeeze(1))
        cls_feat = cls_feat + self.ffn(cls_feat)
        cls_feat = self.norm2(cls_feat)

        return {
            "action_logits":  self.action_head(cls_feat),    # (B, 28)
            "presence_logit": self.presence_head(cls_feat).squeeze(-1),  # (B,)
        }


# ──────────────────────────────────────────────
# CSIEncoder — load từ SSL checkpoint
# ──────────────────────────────────────────────

class CSIEncoder(nn.Module):
    """
    Load SSL checkpoint, output (B, T, N_patches, D) — KHÔNG mean pool.
    """
    def __init__(self, cfg_model: dict):
        super().__init__()
        self.tokenizer = CSITokenizer(270, cfg_model["patch_size"], cfg_model["d_model"])
        self.spatial_encoder = SpatialEncoder(
            cfg_model["d_model"], cfg_model["spatial_heads"],
            cfg_model["n_spatial_layers"], cfg_model.get("ffn_mult", 4), cfg_model.get("dropout", 0.1),
        )
        self.spatial_to_temporal_norm = nn.LayerNorm(cfg_model["d_model"])
        self.temporal_encoder = TemporalEncoder(
            cfg_model["d_model"], cfg_model["temporal_heads"],
            cfg_model["n_temporal_layers"], cfg_model.get("ffn_mult", 4), cfg_model.get("dropout", 0.1),
        )

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, freeze: bool = True) -> "CSIEncoder":
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg   = ckpt["config"]["model"]
        model = cls(cfg)

        # Load tokenizer
        model.tokenizer.norm.load_state_dict(ckpt["tokenizer"], strict=False)
        model.tokenizer.patch_embed.load_state_dict(
            {k.replace("patch_embed.", ""): v
             for k, v in ckpt["tokenizer"].items() if k.startswith("patch_embed.")},
            strict=False,
        )
        model.tokenizer.temporal_pe.copy_(ckpt["tokenizer"]["temporal_pe.pe"])

        # Load encoders
        _load_encoder_layers(model.spatial_encoder.layers,
                             model.spatial_encoder.norm,
                             ckpt["spatial_encoder"])
        _load_encoder_layers(model.temporal_encoder.layers,
                             model.temporal_encoder.norm,
                             ckpt["temporal_encoder"])
        model.spatial_to_temporal_norm.load_state_dict(ckpt["spatial_to_temporal_norm"])

        if freeze:
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
            print(f"[CSIEncoder] FROZEN | val_recon_loss={ckpt.get('val_recon_loss', '?'):.4f}")
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, 60, 270) → (B, 60, 45, 256)"""
        tokens = self.tokenizer(x)
        tokens = self.spatial_encoder(tokens)
        tokens = self.spatial_to_temporal_norm(tokens)
        tokens = self.temporal_encoder(tokens)
        return tokens   # (B, T, N, D) — KHÔNG mean pool


def _load_encoder_layers(layers, norm, state):
    layer_state, norm_state = {}, {}
    for k, v in state.items():
        if k.startswith("layers."):
            parts = k.split(".", 2)
            layer_state.setdefault(int(parts[1]), {})[parts[2]] = v
        elif k.startswith("norm."):
            norm_state[k[5:]] = v
    for i, layer in enumerate(layers):
        if i in layer_state:
            layer.load_state_dict(layer_state[i], strict=False)
    if norm_state:
        norm.load_state_dict(norm_state, strict=False)


class RFPoseModel(nn.Module):
    def __init__(self, encoder, n_joints=13, num_actions=28,
                 d_model=256, dropout=0.1,
                 n_decoder_layers=3, n_temporal_layers=2):
        super().__init__()
        self.encoder      = encoder
        self.pose_decoder = PoseDecoder(n_joints, d_model, n_heads=8,
                                        n_decoder_layers=n_decoder_layers,
                                        n_temporal_layers=n_temporal_layers,
                                        dropout=dropout)
        self.cls_module   = CLSTokenModule(d_model, n_heads=8,
                                           dropout=dropout, num_actions=num_actions)

    @classmethod
    def from_ssl_checkpoint(cls, ssl_ckpt_path: str,
                            n_joints=13, num_actions=28,
                            dropout=0.1, freeze_encoder=True,
                            n_decoder_layers=3, n_temporal_layers=2) -> "RFPoseModel":
        encoder = CSIEncoder.from_checkpoint(ssl_ckpt_path, freeze=freeze_encoder)
        ckpt    = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)
        d_model = ckpt["config"]["model"]["d_model"]

        model = cls(encoder, n_joints, num_actions, d_model, dropout,
                    n_decoder_layers, n_temporal_layers)

        n_total     = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[RFPoseModel] total={n_total:,}  trainable={n_trainable:,}")
        return model

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, 2, 60, 270)
        returns dict:
            coords:         (B, 60, 13, 3)
            vis_logits:     (B, 60, 13)
            action_logits:  (B, 28)
            presence_logit: (B,)
        """
        # Encoder luôn frozen → torch.no_grad() unconditional
        # self.encoder.training luôn False sau from_checkpoint(freeze=True)
        with torch.no_grad():
            features = self.encoder(x)              # (B, T, N, D)

        coords, vis_logits = self.pose_decoder(features)
        cls_out            = self.cls_module(features)

        return {
            "coords":         coords,               # (B, T, 13, 3)
            "vis_logits":     vis_logits,            # (B, T, 13)
            "action_logits":  cls_out["action_logits"],
            "presence_logit": cls_out["presence_logit"],
        }