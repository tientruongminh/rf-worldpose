"""WiMose Masked Auto-Encoder (WiMAE) — Phase 1 pretraining.

Architecture (DT-Pose phase-1 adapted for WiFi CSI):

  CSI image  (B, 2, 342, 60)
      │
  PatchEmbed2D  patch=(18, 6) → 190 tokens of dim 216
      │  linear → embed_dim=384
      │
  Add 2-D sinusoidal position embedding
      │
  Random mask 80 % of tokens  →  keep 38 visible tokens
      │
  ViT Encoder  (6 layers, 6 heads, embed_dim=384)
      │  visible tokens only
      │
  Decoder (4 layers, 4 heads, embed_dim=256)
      │  full sequence: visible tokens + [MASK] tokens
      │  project encoder tokens to decoder dim first
      │
  Pixel-prediction head  →  reconstruct 216-dim patch values
      │
  Loss: MSE on masked patches only

After pretraining, **only the Encoder is kept**.  The encoder can be used as a
drop-in replacement for the ResNet backbone in WiMoseNet by calling
``encode(x)`` which returns (B, embed_dim) after mean-pooling over tokens.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2-D patch embedding for CSI images
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """Split (B, C, H, W) CSI image into non-overlapping patches and project.

    Args:
        in_channels:  Number of input channels (default 2: amp+phase).
        img_h:        Image height  = N_subcarriers (default 342).
        img_w:        Image width   = time window   (default 60).
        patch_h:      Patch height (default 18 → 19 patches along H).
        patch_w:      Patch width  (default  6 → 10 patches along W).
        embed_dim:    Output embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = 2,
        img_h: int = 342,
        img_w: int = 60,
        patch_h: int = 18,
        patch_w: int = 6,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        assert img_h % patch_h == 0 and img_w % patch_w == 0, (
            f"Image ({img_h}×{img_w}) must be divisible by patch ({patch_h}×{patch_w})"
        )
        self.patch_h  = patch_h
        self.patch_w  = patch_w
        self.n_h      = img_h // patch_h           # 19
        self.n_w      = img_w // patch_w           # 10
        self.n_tokens = self.n_h * self.n_w        # 190
        self.patch_dim = in_channels * patch_h * patch_w  # 216

        self.proj = nn.Linear(self.patch_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → tokens (B, N, embed_dim)."""
        B, C, H, W = x.shape
        # reshape into patches: (B, n_h, patch_h, n_w, patch_w, C)
        x = x.view(B, C, self.n_h, self.patch_h, self.n_w, self.patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, n_h, n_w, C, ph, pw)
        x = x.view(B, self.n_tokens, self.patch_dim)    # (B, N, patch_dim)
        return self.proj(x)                              # (B, N, embed_dim)


# ---------------------------------------------------------------------------
# Sinusoidal 2-D positional embedding
# ---------------------------------------------------------------------------

def _sincos_2d(n_h: int, n_w: int, embed_dim: int) -> torch.Tensor:
    """Return (n_h*n_w, embed_dim) 2-D sinusoidal position embedding.

    Half of embed_dim encodes row position, half encodes column position.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2-D sin/cos"
    d = embed_dim // 4
    freq = 1.0 / (10000 ** (torch.arange(0, d, dtype=torch.float32) / d))

    row_idx = torch.arange(n_h, dtype=torch.float32)  # (n_h,)
    col_idx = torch.arange(n_w, dtype=torch.float32)  # (n_w,)

    row_sin = torch.sin(row_idx.unsqueeze(1) * freq.unsqueeze(0))   # (n_h, d)
    row_cos = torch.cos(row_idx.unsqueeze(1) * freq.unsqueeze(0))
    col_sin = torch.sin(col_idx.unsqueeze(1) * freq.unsqueeze(0))   # (n_w, d)
    col_cos = torch.cos(col_idx.unsqueeze(1) * freq.unsqueeze(0))

    # Broadcast: each of the n_h*n_w positions gets row+col encoding
    row_emb = torch.cat([row_sin, row_cos], dim=-1)  # (n_h, 2d)
    col_emb = torch.cat([col_sin, col_cos], dim=-1)  # (n_w, 2d)

    grid_row = row_emb.unsqueeze(1).expand(n_h, n_w, 2 * d)  # (n_h, n_w, 2d)
    grid_col = col_emb.unsqueeze(0).expand(n_h, n_w, 2 * d)  # (n_h, n_w, 2d)

    pe = torch.cat([grid_row, grid_col], dim=-1)         # (n_h, n_w, embed_dim)
    return pe.view(n_h * n_w, embed_dim)                 # (N, embed_dim)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + out)


class _FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = _Attention(dim, n_heads, dropout)
        self.ffn  = _FFN(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x))


# ---------------------------------------------------------------------------
# Encoder  (kept after pretraining for fine-tuning)
# ---------------------------------------------------------------------------

class WiMAEEncoder(nn.Module):
    """ViT-style Transformer encoder for WiFi CSI images.

    Usable both during MAE pretraining (on visible tokens only) and during
    fine-tuning / inference (on full, unmasked token sequence).

    Args:
        in_channels:  CSI channels (default 2).
        img_h, img_w: CSI image dimensions.
        patch_h, patch_w: Patch sizes.
        embed_dim:    Token embedding dimension (default 384).
        n_layers:     Transformer depth (default 6).
        n_heads:      Number of attention heads (default 6).
        mlp_ratio:    FFN hidden ratio (default 4).
        dropout:      Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 2,
        img_h: int = 342,
        img_w: int = 60,
        patch_h: int = 18,
        patch_w: int = 6,
        embed_dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(in_channels, img_h, img_w, patch_h, patch_w, embed_dim)
        self.embed_dim   = embed_dim
        n_tokens = self.patch_embed.n_tokens

        # Fixed sinusoidal positional embedding — not a learned parameter
        pe = _sincos_2d(self.patch_embed.n_h, self.patch_embed.n_w, embed_dim)
        self.register_buffer("pos_embed", pe.unsqueeze(0))  # (1, N, D)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:          (B, C, H, W) — full CSI image.
            token_mask: (B, N) bool tensor — True means *visible* (kept).
                        If None, all tokens are processed (inference mode).
        Returns:
            (B, N_vis, D) if token_mask given, else (B, N, D).
        """
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)

        if token_mask is not None:
            # Gather only the visible tokens for each sample.
            # token_mask: (B, N) bool, True = keep
            B, N, D = tokens.shape
            # stack visible tokens; requires same n_visible per batch item
            n_vis = token_mask[0].sum().item()
            visible = tokens[token_mask.bool()].view(B, n_vis, D)
            tokens = visible

        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full-image inference: returns mean-pooled feature (B, embed_dim)."""
        tokens = self.forward(x, token_mask=None)  # (B, N, D)
        return tokens.mean(dim=1)                  # (B, D)


# ---------------------------------------------------------------------------
# Decoder  (used only during pretraining; discarded afterwards)
# ---------------------------------------------------------------------------

class WiMAEDecoder(nn.Module):
    """Light-weight Transformer decoder that reconstructs masked patches.

    Args:
        encoder_dim:  Encoder embedding dimension (input projection source).
        decoder_dim:  Decoder embedding dimension (default 256).
        n_tokens:     Total number of tokens (190).
        n_layers:     Decoder depth (default 4).
        n_heads:      Number of attention heads (default 4).
        patch_dim:    Patch pixel dimension to reconstruct (default 216).
        n_h, n_w:     Grid dimensions for positional embedding.
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        decoder_dim: int = 256,
        n_tokens: int = 190,
        n_layers: int = 4,
        n_heads: int = 4,
        patch_dim: int = 216,
        n_h: int = 19,
        n_w: int = 10,
    ) -> None:
        super().__init__()
        self.n_tokens   = n_tokens
        self.decoder_dim = decoder_dim

        # Project encoder tokens into decoder space
        self.enc2dec = nn.Linear(encoder_dim, decoder_dim, bias=True)

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Fixed sinusoidal positional embedding for decoder (full sequence)
        pe = _sincos_2d(n_h, n_w, decoder_dim)
        self.register_buffer("pos_embed", pe.unsqueeze(0))  # (1, N, D)

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, n_heads)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_dim)  # pixel predictor

    def forward(
        self,
        encoder_out: torch.Tensor,   # (B, N_vis, encoder_dim) — visible token features
        token_mask: torch.Tensor,    # (B, N_total) bool — True = visible
    ) -> torch.Tensor:
        """Returns reconstructed patch values (B, N_total, patch_dim)."""
        B, N_vis, _ = encoder_out.shape
        N_total = token_mask.shape[1]

        # Project visible encoder tokens
        vis_tokens = self.enc2dec(encoder_out)  # (B, N_vis, D)

        # Build full-sequence decoder input: fill [MASK] for masked positions.
        # Under AMP, vis_tokens may be fp16 while mask_token/pos_embed buffers are fp32.
        dtype = vis_tokens.dtype
        full_tokens = self.mask_token.expand(B, N_total, -1).to(dtype=dtype).clone()
        vis_idx = token_mask.bool()  # (B, N)
        full_tokens[vis_idx] = vis_tokens.reshape(B * N_vis, -1)

        # Add positional embedding (cast buffer to activation dtype)
        full_tokens = full_tokens + self.pos_embed.to(dtype=dtype)

        for blk in self.blocks:
            full_tokens = blk(full_tokens)
        full_tokens = self.norm(full_tokens)
        return self.head(full_tokens)  # (B, N_total, patch_dim)


# ---------------------------------------------------------------------------
# Full MAE model
# ---------------------------------------------------------------------------

class WiMoseMAE(nn.Module):
    """Full WiFi CSI Masked Auto-Encoder.

    During forward():  applies random masking, encodes visible tokens,
                       decodes all tokens, computes MSE loss on masked patches.
    After pretraining: use encoder.encode(x) to get (B, embed_dim) WiFi feature.

    Args:
        mask_ratio:   Fraction of tokens to mask (default 0.80 per MAE paper).
        encoder / decoder kwargs passed through.
    """

    def __init__(
        self,
        mask_ratio: float = 0.80,
        in_channels: int = 2,
        img_h: int = 342,
        img_w: int = 60,
        patch_h: int = 18,
        patch_w: int = 6,
        # encoder
        encoder_dim: int = 384,
        encoder_layers: int = 6,
        encoder_heads: int = 6,
        # decoder
        decoder_dim: int = 256,
        decoder_layers: int = 4,
        decoder_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

        self.encoder = WiMAEEncoder(
            in_channels=in_channels,
            img_h=img_h, img_w=img_w,
            patch_h=patch_h, patch_w=patch_w,
            embed_dim=encoder_dim,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            dropout=dropout,
        )

        n_tokens  = self.encoder.patch_embed.n_tokens    # 190
        patch_dim = self.encoder.patch_embed.patch_dim   # 216
        n_h       = self.encoder.patch_embed.n_h         # 19
        n_w       = self.encoder.patch_embed.n_w         # 10

        self.decoder = WiMAEDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            n_tokens=n_tokens,
            n_layers=decoder_layers,
            n_heads=decoder_heads,
            patch_dim=patch_dim,
            n_h=n_h,
            n_w=n_w,
        )

        # Store patch shape for loss computation
        self._in_ch    = in_channels
        self._patch_h  = patch_h
        self._patch_w  = patch_w
        self._n_h      = n_h
        self._n_w      = n_w
        self._n_tokens = n_tokens
        self._patch_dim = patch_dim

    # ── masking helpers ────────────────────────────────────────────────────

    def _random_mask(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Returns (B, N) bool mask: True = visible (kept), False = masked.

        Generates the same mask for each batch item independently (no block masking).
        """
        n_vis = max(1, int(N * (1.0 - self.mask_ratio)))
        rand  = torch.rand(B, N, device=device)
        ids   = rand.argsort(dim=1)          # (B, N) — random permutation
        mask  = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :n_vis], True)   # mark n_vis as visible
        return mask  # True = visible

    # ── reconstruct target (ground-truth patch pixels) ────────────────────

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, N, patch_dim) — the reconstruction target."""
        B, C, H, W = x.shape
        x = x.view(B, C, self._n_h, self._patch_h, self._n_w, self._patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, self._n_tokens, self._patch_dim)

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 2, 342, 60) CSI image.
        Returns:
            dict with keys:
              "loss"      — scalar reconstruction loss (masked MSE)
              "pred"      — (B, N, patch_dim) all-token predictions
              "mask"      — (B, N) bool, True = visible
        """
        B = x.shape[0]
        N = self._n_tokens
        token_mask = self._random_mask(B, N, x.device)   # (B, N) True=visible

        enc_out = self.encoder(x, token_mask=token_mask)  # (B, N_vis, D)
        pred    = self.decoder(enc_out, token_mask)        # (B, N, patch_dim)

        target  = self._patchify(x)  # (B, N, patch_dim) — original patches

        # Loss only on masked tokens (compute in fp32 for AMP stability)
        masked_pred   = pred[~token_mask]    # (M, patch_dim)
        masked_target = target[~token_mask]  # (M, patch_dim)
        loss = F.mse_loss(masked_pred.float(), masked_target.float())

        return {"loss": loss, "pred": pred, "mask": token_mask}
