"""
csi_tokenizer.py
----------------
Tokenizes raw WiFi CSI data thành patch embeddings cho Transformer.

Input CSI shape từ ESP32-S3 (802.11n HT40):
    (B, T, N_sub, 2)
    B       = batch size
    T       = số time steps (sliding window, e.g. 100 frames @ 100Hz = 1 giây)
    N_sub   = số subcarrier = 114 (HT40) hoặc 52 (HT20)
    2       = [amplitude, phase] đã được decode từ Bronze layer

Pipeline tokenize:
    1. Amplitude + phase normalization (per-subcarrier running stats)
    2. Subcarrier patching:  N_sub -> N_patches = N_sub // patch_size
    3. Linear projection mỗi patch -> d_model
    4. Positional embedding (learnable) cho subcarrier axis
    5. Temporal CLS token cho mỗi time step (optional, dùng khi pool theo thời gian)
    6. Temporal positional embedding cho T axis
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # pip install einops


# ---------------------------------------------------------------------------
# 1. Per-subcarrier normalization (chạy online, không cần fit trước)
# ---------------------------------------------------------------------------
class CSIRunningNorm(nn.Module):
    """
    Running mean/std normalization theo subcarrier axis.
    Dùng trong inference để không cần biết global stats trước.
    Momentum = 0.01 => cập nhật chậm, ổn định hơn BatchNorm.
    """

    def __init__(self, n_subcarriers: int = 114, momentum: float = 0.01, eps: float = 1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        # Buffer (không phải parameter, không update bằng grad)
        self.register_buffer("running_mean", torch.zeros(n_subcarriers, 2))  # [amp, phase]
        self.register_buffer("running_var", torch.ones(n_subcarriers, 2))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_sub, 2)
        returns: normalized x, cùng shape
        """
        if self.training:
            # Tính mean/var trên B và T axis => shape (N_sub, 2)
            mean = x.mean(dim=(0, 1))          # (N_sub, 2)
            var  = x.var(dim=(0, 1), unbiased=False)

            if not self.initialized:
                self.running_mean.copy_(mean)
                self.running_var.copy_(var)
                self.initialized.fill_(True)
            else:
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)

        mean = self.running_mean  # (N_sub, 2)
        std  = (self.running_var + self.eps).sqrt()

        # Broadcast: (1, 1, N_sub, 2)
        return (x - mean.unsqueeze(0).unsqueeze(0)) / std.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# 2. Subcarrier Patch Embedding
# ---------------------------------------------------------------------------
class SubcarrierPatchEmbed(nn.Module):
    """
    Chia N_sub subcarriers thành N_patches = N_sub // patch_size nhóm,
    mỗi nhóm (patch_size * 2 features) -> d_model qua Linear.

    Tại sao patch?
    - Subcarrier liền kề mang thông tin tương quan (frequency coherence)
    - Patch giảm sequence length => attention rẻ hơn O(T^2)
    - Tương tự Vision Transformer nhưng trên frequency axis thay vì spatial

    Args:
        n_subcarriers: số subcarrier (default 114 cho ESP32-S3 HT40)
        patch_size:    số subcarrier per patch (default 6 => 19 patches)
        d_model:       output embedding dim
        dropout:       dropout sau projection
    """

    def __init__(
        self,
        n_subcarriers: int = 114,
        patch_size: int = 6,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_subcarriers % patch_size == 0, (
            f"n_subcarriers ({n_subcarriers}) phải chia hết cho patch_size ({patch_size})"
        )
        self.patch_size = patch_size
        self.n_patches = n_subcarriers // patch_size  # 114 // 6 = 19
        patch_dim = patch_size * 2  # amplitude + phase cho mỗi subcarrier trong patch

        # Linear projection
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional embedding theo subcarrier (spatial freq position)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_sub, 2)
        returns: (B, T, N_patches, d_model)
        """
        B, T, N, C = x.shape  # C = 2

        # Gộp [amp, phase] thành một vector rồi nhóm theo patch_size
        # (B, T, N, 2) -> (B, T, N_patches, patch_size*2)
        x = rearrange(x, "b t (np ps) c -> b t np (ps c)", ps=self.patch_size)

        # Project -> (B, T, N_patches, d_model)
        x = self.proj(x)

        # Cộng positional embedding (broadcast theo B và T)
        x = x + self.pos_embed

        return self.dropout(x)


# ---------------------------------------------------------------------------
# 3. Temporal Positional Embedding
# ---------------------------------------------------------------------------
class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal encoding cho time axis T.
    Dùng sinusoidal thay vì learnable vì T có thể thay đổi lúc inference
    (variable-length sequence).
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, 1, d_model) để broadcast với (B, T, N_patches, d_model)
        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_patches, d_model)
        """
        T = x.size(1)
        return self.dropout(x + self.pe[:, :T, :, :])


# ---------------------------------------------------------------------------
# 4. Multi-Node Fusion (optional - nếu có nhiều ESP32-S3)
# ---------------------------------------------------------------------------
class MultiNodeFusion(nn.Module):
    """
    RF-WorldPose dùng 4 ESP32-S3 nodes đặt ở 4 góc phòng.
    Module này fuse CSI từ nhiều node trước khi tokenize.

    Strategy: cross-node attention (mỗi node là một "view")
    Input:  list of (B, T, N_sub, 2) — một tensor mỗi node
    Output: (B, T, N_sub, 2) — fused

    Hoặc đơn giản hơn: concatenate trên subcarrier axis
    """

    def __init__(self, n_nodes: int = 4, n_subcarriers: int = 114, d_model: int = 256):
        super().__init__()
        self.n_nodes = n_nodes
        # Node embedding để phân biệt từng sensor
        self.node_embed = nn.Embedding(n_nodes, n_subcarriers * 2)  # (N_node, N_sub*2)

        # Attention pooling qua nodes
        self.node_attn = nn.MultiheadAttention(
            embed_dim=n_subcarriers * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(n_subcarriers * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_nodes, T, N_sub, 2)  — Tensor 5D từ DataLoader
        returns: (B, T, N_sub, 2) — fused
        """
        # Đổi trục để đưa n_nodes về vị trí xử lý: [B, T, n_nodes, N_sub, 2]
        x = x.transpose(1, 2)
        B, T, n_nodes, N, C = x.shape

        # Flatten subcarrier + channel: [B, T, n_nodes, N*C]
        stacked = x.reshape(B, T, n_nodes, N * C)

        # Cộng node positional embedding
        node_ids = torch.arange(self.n_nodes, device=stacked.device)
        node_pe = self.node_embed(node_ids).unsqueeze(0).unsqueeze(0)  # (1, 1, n_nodes, N*C)
        stacked = stacked + node_pe

        # Flatten B,T để attention hoạt động: (B*T, n_nodes, N*C)
        stacked = rearrange(stacked, "b t n d -> (b t) n d")

        # Self-attention qua nodes (mỗi node attend các node khác)
        fused, _ = self.node_attn(stacked, stacked, stacked)
        fused = self.norm(fused + stacked)  # residual

        # Mean pool qua nodes -> (B*T, N*C) -> (B, T, N, C)
        fused = fused.mean(dim=1)  # (B*T, N*C)
        fused = rearrange(fused, "(b t) (n c) -> b t n c", b=B, t=T, n=N, c=C)

        return fused


# ---------------------------------------------------------------------------
# 5. CSITokenizer — tổng hợp tất cả
# ---------------------------------------------------------------------------
class CSITokenizer(nn.Module):
    """
    Full tokenization pipeline: raw CSI -> token sequence cho Transformer.

    Luồng:
        raw CSI (B, T, N_sub, 2)
           -> CSIRunningNorm          (normalize per-subcarrier)
           -> SubcarrierPatchEmbed    (B, T, N_patches, d_model)
           -> TemporalPositionalEnc   (+ temporal position info)
           -> output: (B, T, N_patches, d_model)

    Nếu dùng multi-node: MultiNodeFusion trước normalize.

    Args:
        n_subcarriers: 114 (HT40) hoặc 52 (HT20)
        patch_size:    subcarrier per patch, phải là ước của n_subcarriers
        d_model:       embedding dimension
        max_seq_len:   max temporal length T
        n_nodes:       số ESP32 nodes (1 = single node, 4 = multi-node)
        dropout:       dropout rate
    """

    def __init__(
        self,
        n_subcarriers: int = 114,
        patch_size: int = 6,
        d_model: int = 256,
        max_seq_len: int = 500,
        n_nodes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_patches = n_subcarriers // patch_size

        # Multi-node fusion (bỏ qua nếu single node)
        if n_nodes > 1:
            self.node_fusion = MultiNodeFusion(n_nodes, n_subcarriers, d_model)
        else:
            self.node_fusion = None

        self.norm       = CSIRunningNorm(n_subcarriers)
        self.patch_embed = SubcarrierPatchEmbed(n_subcarriers, patch_size, d_model, dropout)
        self.temporal_pe = TemporalPositionalEncoding(d_model, max_seq_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_sub, 2)               — single node (4D)
           hoặc (B, n_nodes, T, N_sub, 2) — multi-node (5D), compatible với DataLoader
        returns: (B, T, N_patches, d_model)
        """
        # Multi-node fusion nếu cần
        if self.node_fusion is not None:
            assert x.ndim == 5, (
                f"Multi-node mode yêu cầu Tensor 5D [B, n_nodes, T, N_sub, 2], "
                f"nhận được shape: {tuple(x.shape)}"
            )
            x = self.node_fusion(x)

        # Normalize
        x = self.norm(x)           # (B, T, N_sub, 2)

        # Patch embed (subcarrier axis)
        x = self.patch_embed(x)    # (B, T, N_patches, d_model)

        # Temporal positional encoding
        x = self.temporal_pe(x)    # (B, T, N_patches, d_model)

        return x

    @property
    def output_shape_info(self) -> dict:
        return {
            "shape": "(B, T, N_patches, d_model)",
            "n_patches": self.n_patches,
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Single-node test
    tokenizer = CSITokenizer(n_subcarriers=114, patch_size=6, d_model=256, n_nodes=1)
    x = torch.randn(4, 100, 114, 2)  # batch=4, T=100 frames, 114 subcarriers
    out = tokenizer(x)
    print(f"[Single node] Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
    # Expected: (4, 100, 19, 256)

    # Multi-node test (4 ESP32-S3) — dùng Tensor 5D như DataLoader thực tế trả về
    tokenizer_multi = CSITokenizer(n_subcarriers=114, patch_size=6, d_model=256, n_nodes=4)
    x_multi = torch.randn(4, 4, 100, 114, 2)  # [B, n_nodes, T, N_sub, 2]
    out_multi = tokenizer_multi(x_multi)
    print(f"[Multi node]  Input: {tuple(x_multi.shape)} -> Output: {tuple(out_multi.shape)}")