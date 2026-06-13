"""WiMose Diffusion Decoder — Phase 2 pose estimation.

Architecture: WiFi-conditioned DDPM operating in joint-coordinate space.

  ┌────────────────────────────────────────────────────────────┐
  │  WiFi CSI (B, 2, 342, 60)                                   │
  │       │                                                      │
  │  WiMAEEncoder (frozen or fine-tuned)                        │
  │       │  encode() → mean-pool → (B, enc_dim=384)             │
  │       │                                                      │
  │  WiFi condition  z_wifi ∈ R^{384}                           │
  └──────────────────────────┬─────────────────────────────────┘
                             │ cross-attention / FiLM
  Pose noise  x_T ~ N(0, I) │
     (B, J, 3)               │
       │                     │
  ┌────┴─────────────────────┴──────────────────────────────┐
  │  DiT  — Diffusion Transformer Decoder                    │
  │                                                          │
  │  • Project joints to token dim:  (B, J, d_model)        │
  │  • Add timestep embedding (sinusoidal + MLP → FiLM)     │
  │  • Add WiFi conditioning (linear proj → FiLM)           │
  │  • N × DiTBlock:                                        │
  │      - Self-attention over joint tokens                  │
  │      - FiLM modulation from timestep + WiFi condition   │
  │      - FFN                                               │
  │  • Predict noise ε̂ (B, J, 3)                            │
  └──────────────────────────────────────────────────────────┘

Training: standard DDPM ε-prediction
  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0,I)
  L = ‖ε - model(x_t, t, z_wifi)‖²

Inference (DDIM, fast 20 steps):
  Sample N hypotheses by running N independent denoising chains
  from different starting noises → diverse, plausible 3D poses.

Reference: Ho et al. DDPM NeurIPS 2020; Song et al. DDIM ICLR 2021;
           Pham et al. DT-Pose ICLR 2025.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Diffusion noise schedule
# ---------------------------------------------------------------------------

class LinearNoiseSchedule:
    """Linear β schedule as in Ho et al. 2020."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)  # (T,)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)          # ᾱ_t

        self.register = {}
        for name, val in [
            ("betas", betas),
            ("alphas", alphas),
            ("alphas_bar", alphas_bar),
            ("sqrt_alphas_bar", alphas_bar.sqrt()),
            ("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt()),
            ("alphas_bar_prev", torch.cat([torch.ones(1), alphas_bar[:-1]])),
        ]:
            self.register[name] = val

    def to(self, device: torch.device) -> "LinearNoiseSchedule":
        self.register = {k: v.to(device) for k, v in self.register.items()}
        return self

    def q_sample(
        self,
        x0: torch.Tensor,  # (B, J, 3)
        t: torch.Tensor,   # (B,) long
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t from x_0."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab   = self.register["sqrt_alphas_bar"][t].view(-1, 1, 1)
        sqrt_1mab = self.register["sqrt_one_minus_alphas_bar"][t].view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn,               # callable: (x_t, t, cond) → noise_pred (B, J, 3)
        cond: torch.Tensor,     # (B, enc_dim)
        shape: tuple[int, ...], # (B, J, 3)
        n_steps: int = 20,
        eta: float = 0.0,       # 0 = deterministic DDIM; 1 = stochastic DDPM
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """DDIM reverse diffusion. Returns denoised x_0 ∈ R^{J×3}."""
        x = torch.randn(shape, device=device)

        # Uniform timestep subsequence
        step = self.T // n_steps
        ts   = list(reversed(range(0, self.T, step)))  # [T-step, ..., 0]

        for i, t_cur in enumerate(ts):
            t_tensor = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            noise_pred = model_fn(x, t_tensor, cond)

            ab  = self.register["sqrt_alphas_bar"][t_cur] ** 2
            x0_pred = (x - (1 - ab).sqrt() * noise_pred) / ab.sqrt()
            x0_pred = x0_pred.clamp(-5, 5)

            if i < len(ts) - 1:
                t_next = ts[i + 1]
                ab_next = self.register["sqrt_alphas_bar"][t_next] ** 2
                sigma   = eta * ((1 - ab_next) / (1 - ab) * (1 - ab / ab_next)).sqrt()
                x = ab_next.sqrt() * x0_pred \
                    + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * noise_pred \
                    + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_timestep_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) long → (B, dim) sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb  = torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)
    return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal + 2-layer MLP → scalar scale/shift for FiLM."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim * 2),  # → (scale, shift)
        )

    def forward(self, t: torch.Tensor, sin_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb = sinusoidal_timestep_embed(t, sin_dim)  # (B, sin_dim)
        ss  = self.mlp(emb)                           # (B, out_dim*2)
        scale, shift = ss.chunk(2, dim=-1)            # (B, out_dim) each
        return scale, shift


# ---------------------------------------------------------------------------
# DiT block (Diffusion Transformer block)
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Self-attention + FiLM conditioning from timestep + WiFi features.

    FiLM: out = scale · LayerNorm(x) + shift
    where (scale, shift) come from timestep and WiFi condition projections.
    """

    def __init__(self, d_model: int, n_heads: int, cond_dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        # Condition projection: (B, cond_dim) → (B, d_model*2) for scale+shift
        # Applied TWICE: once for attn, once for ffn
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, d_model * 4),   # → (sa_scale, sa_shift, ffn_scale, ffn_shift)
        )

    def forward(
        self,
        x:    torch.Tensor,   # (B, J, d_model)
        cond: torch.Tensor,   # (B, cond_dim) — timestep + WiFi fused condition
    ) -> torch.Tensor:
        # condition → 4 vectors
        c = self.cond_proj(cond)                # (B, 4*d_model)
        sa_scale, sa_shift, ff_scale, ff_shift = c.chunk(4, dim=-1)  # each (B, d_model)

        # Self-attention with FiLM
        x_norm = self.norm1(x) * (1 + sa_scale.unsqueeze(1)) + sa_shift.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        # FFN with FiLM
        x_norm = self.norm2(x) * (1 + ff_scale.unsqueeze(1)) + ff_shift.unsqueeze(1)
        x = x + self.ffn(x_norm)

        return x


# ---------------------------------------------------------------------------
# Diffusion noise predictor (DiT denoiser)
# ---------------------------------------------------------------------------

class WiMoseDiT(nn.Module):
    """Diffusion Transformer conditioned on WiFi features.

    Predicts the noise ε given noisy pose x_t, timestep t, and WiFi feature z.

    Args:
        n_joints:     Number of skeleton joints (17 for H36M).
        coord_dim:    Coordinates per joint (default 3).
        d_model:      Token embedding dimension (default 256).
        n_layers:     Number of DiT blocks (default 8).
        n_heads:      Attention heads (default 8).
        wifi_dim:     WiFi feature dimension from encoder (default 384).
        t_embed_dim:  Timestep sinusoidal dimension (default 256).
    """

    def __init__(
        self,
        n_joints: int = 17,
        coord_dim: int = 3,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        wifi_dim: int = 384,
        t_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.n_joints  = n_joints
        self.d_model   = d_model

        # Project joint coords to token dim
        self.joint_in  = nn.Linear(coord_dim, d_model)

        # Learnable positional embedding per joint
        self.joint_pos = nn.Parameter(torch.randn(1, n_joints, d_model) * 0.02)

        # Timestep embedding
        self.t_embed = TimestepEmbedding(t_embed_dim, d_model)

        # WiFi conditioning: project WiFi feature to d_model
        self.wifi_proj = nn.Linear(wifi_dim, d_model)

        # DiT blocks
        cond_dim = d_model * 2  # timestep scale+shift fused with WiFi feature
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, cond_dim=d_model)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Linear(d_model, coord_dim)

        # Zero-init final projection (stable training per DiT paper)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        x_t:  torch.Tensor,  # (B, J, 3) — noisy pose at timestep t
        t:    torch.Tensor,   # (B,) long  — timestep indices
        z_wifi: torch.Tensor, # (B, wifi_dim) — WiFi feature condition
    ) -> torch.Tensor:
        """Returns predicted noise ε̂ (B, J, 3)."""
        # ── joint token embedding ────────────────────────────────────────────
        tokens = self.joint_in(x_t) + self.joint_pos  # (B, J, d_model)

        # ── condition fusion: timestep + WiFi ────────────────────────────────
        t_scale, t_shift = self.t_embed(t, self.d_model)     # (B, d_model) each
        wifi_feat        = self.wifi_proj(z_wifi)             # (B, d_model)
        # Fuse: timestep modulates WiFi feature, resulting in a single cond vector
        cond = wifi_feat * (1 + t_scale) + t_shift            # (B, d_model)

        # ── DiT blocks ───────────────────────────────────────────────────────
        for blk in self.blocks:
            tokens = blk(tokens, cond)

        # ── output ───────────────────────────────────────────────────────────
        tokens = self.final_norm(tokens)
        return self.final_proj(tokens)  # (B, J, 3) — predicted noise


# ---------------------------------------------------------------------------
# Full model: encoder (WiFi → feature) + DiT denoiser
# ---------------------------------------------------------------------------

class WiMoseDiffNet(nn.Module):
    """End-to-end WiFi pose estimation with diffusion decoder.

    Two operating modes:
    1. *Pretrained encoder*: load encoder weights from MAE pretraining, optionally
       freeze for the first K epochs of diffusion training.
    2. *Scratch*: encoder trained jointly with diffusion decoder (slower convergence).

    Args:
        n_joints:       Skeleton joints (default 17).
        wifi_dim:       WiFi encoder output dim (default 384).
        encoder_cfg:    Dict of kwargs passed to WiMAEEncoder.
                        If None, a WiMoseNet ResNet backbone is used instead.
        dit_cfg:        Dict of kwargs passed to WiMoseDiT.
        noise_schedule: Pre-built LinearNoiseSchedule (created in training).
        freeze_encoder_epochs: First N epochs to freeze encoder (default 0 = joint).
    """

    def __init__(
        self,
        n_joints: int = 17,
        wifi_dim: int = 384,
        dit_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        from rfpose.models.wimose_mae import WiMAEEncoder

        # Default ViT encoder (same as MAE)
        self.encoder = WiMAEEncoder(
            in_channels=2,
            img_h=342, img_w=60,
            patch_h=18, patch_w=6,
            embed_dim=wifi_dim,
            n_layers=6, n_heads=6,
            dropout=0.1,
        )

        d = dit_cfg or {}
        self.denoiser = WiMoseDiT(
            n_joints=n_joints,
            wifi_dim=wifi_dim,
            d_model=d.get("d_model", 256),
            n_layers=d.get("n_layers", 8),
            n_heads=d.get("n_heads", 8),
            t_embed_dim=d.get("t_embed_dim", 256),
        )

    def get_wifi_feature(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, H, W) → (B, wifi_dim) WiFi condition vector."""
        return self.encoder.encode(x)  # mean-pool over tokens

    def forward(
        self,
        x:    torch.Tensor,   # (B, 2, N_sub, T) CSI
        x_t:  torch.Tensor,   # (B, J, 3) noisy pose
        t:    torch.Tensor,   # (B,) timestep
    ) -> torch.Tensor:
        """Returns noise prediction ε̂ (B, J, 3) for training."""
        z_wifi = self.get_wifi_feature(x)           # (B, wifi_dim)
        return self.denoiser(x_t, t, z_wifi)        # (B, J, 3)

    @torch.no_grad()
    def sample(
        self,
        x:          torch.Tensor,   # (B, 2, N_sub, T) CSI
        schedule:   "LinearNoiseSchedule",
        n_steps:    int = 20,
        n_hypotheses: int = 1,
        eta:        float = 0.0,
    ) -> torch.Tensor:
        """Generate pose hypotheses via DDIM reverse diffusion.

        Args:
            n_hypotheses: Number of independent samples (diversity).
                          If > 1, returns (B*n_hypotheses, J, 3) poses.
        Returns:
            (B * n_hypotheses, J, 3) — denoised pose coordinates.
        """
        z_wifi = self.get_wifi_feature(x)           # (B, wifi_dim)
        B, J   = z_wifi.shape[0], self.denoiser.n_joints
        device = x.device

        if n_hypotheses > 1:
            # Repeat WiFi condition for each hypothesis
            z_wifi = z_wifi.repeat_interleave(n_hypotheses, dim=0)  # (B*H, wifi_dim)
            B_eff  = B * n_hypotheses
        else:
            B_eff  = B

        def _model_fn(xt, ts, cond):
            return self.denoiser(xt, ts, cond)

        poses = schedule.ddim_sample(
            model_fn=_model_fn,
            cond=z_wifi,
            shape=(B_eff, J, 3),
            n_steps=n_steps,
            eta=eta,
            device=device,
        )
        return poses  # (B_eff, J, 3)
