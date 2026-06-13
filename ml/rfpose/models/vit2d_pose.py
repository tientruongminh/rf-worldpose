"""
vit2d_pose.py
"""
import torch
import torch.nn as nn
from einops import rearrange

# Re-use the team's excellent decoders to ensure 100% output compatibility
from rfpose.models.transformer import PoseDecoder, CLSTokenModule

class CsiPatchEmbedding2D(nn.Module):
    """
    Extracts 2D patches from CSI (Subcarrier x Time) using Conv2d.
    Uses padding and stride=1 on the time axis to preserve the exact T=60 
    resolution required by the team's PoseDecoder.
    """
    def __init__(self, in_channels: int = 2, d_model: int = 256, patch_freq: int = 6, patch_time: int = 3):
        super().__init__()
        patch_freq = int(patch_freq)
        patch_time = int(patch_time)
        self.patch_freq = patch_freq
        
        # Stride=(patch_freq, 1) ensures we compress frequency but preserve every time frame.
        # Padding on time axis ensures the output sequence length remains T.
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=(patch_freq, patch_time),
            stride=(patch_freq, 1),
            padding=(0, patch_time // 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, T, N_sub, 2]
        # Target shape for Conv2d: [B, Channels, Height(Freq), Width(Time)]
        x = x.permute(0, 3, 2, 1).contiguous()
        
        # Apply 2D convolution
        z = self.proj(x) # [B, d_model, N_patches, T]
        
        # Trim extra time frame if padding added one (due to even/odd math)
        z = z[:, :, :, :x.shape[-1]]
        
        # Reshape back to Transformer standard: [B, T, N_patches, d_model]
        z = z.permute(0, 3, 2, 1).contiguous()
        return z


class CSIViT2DPose(nn.Module):
    """
    Custom 2D Vision Transformer for CSI Pose Estimation.
    Treats WiFi signals as a spatio-temporal image.
    """
    def __init__(
        self, 
        n_subcarriers: int = 114,
        patch_freq: int = 6,
        d_model: int = 256, 
        n_layers: int = 4, 
        n_heads: int = 8,
        n_joints: int = 17,
        num_actions: int = 28
    ):
        super().__init__()
        patch_freq = int(patch_freq)
        n_subcarriers = int(n_subcarriers)
        self.d_model = d_model
        self.n_patches = n_subcarriers // patch_freq
        
        # 1. 2D Tokenizer
        self.tokenizer = CsiPatchEmbedding2D(d_model=d_model, patch_freq=patch_freq, patch_time=3)
        
        # 2. Positional Embeddings (Spatial & Temporal)
        self.spatial_pe = nn.Parameter(torch.zeros(1, 1, self.n_patches, d_model))
        self.temporal_pe = nn.Parameter(torch.zeros(1, 500, 1, d_model)) # Max 500 frames
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        
        # 3. Standard ViT Encoder (Processes flat 2D patches globally)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            dropout=0.1, batch_first=True
        )
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Decoders (Inherited from team's architecture)
        self.cls_module = CLSTokenModule(
            d_model=d_model, n_heads=n_heads, n_patches=self.n_patches, num_actions=num_actions
        )
        self.pose_decoder = PoseDecoder(
            n_joints=n_joints, d_model=d_model, n_heads=n_heads, 
            n_decoder_layers=3, n_temporal_layers=2
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> dict:
        # Step 1: 2D Patching -> [B, T, N_patches, D]
        tokens = self.tokenizer(x)
        B, T, N, D = tokens.shape
        
        # Step 2: Add Positional Encodings
        tokens = tokens + self.spatial_pe[:, :, :N, :]
        tokens = tokens + self.temporal_pe[:, :T, :, :].expand(-1, -1, N, -1)
        
        # Step 3: Flatten for ViT Encoder -> [B, T*N, D]
        flat_tokens = rearrange(tokens, "b t n d -> b (t n) d")
        
        # ViT processing (Global spatio-temporal attention)
        encoded_flat = self.vit_encoder(flat_tokens)
        
        # Unflatten back to [B, T, N, D] for the decoders
        encoded_features = rearrange(encoded_flat, "b (t n) d -> b t n d", b=B, t=T, n=N)
        
        # Step 4: Action Recognition via CLS Module
        # Note: CLS module handles the flatten internally
        cls_out = self.cls_module(encoded_features)
        
        # Step 5: Pose Estimation (Root-relative prediction)
        coords, vis_logits = self.pose_decoder(encoded_features)
        
        # Return exact dictionary format expected by RFPoseLoss
        return {
            "coords": coords,
            "vis_logits": vis_logits,
            "action_logits": cls_out["action_logits"],
            "presence_logit": cls_out["presence_logit"],
            "cls_feat": cls_out["cls_feat"],
            "spatial_feat": encoded_features, # Dummy pass-through for loss compat
            "temporal_feat": encoded_features # Dummy pass-through for loss compat
        }