"""
losses.py
---------
Loss functions cho WiFi CSI -> Pose Estimation.

Tổng hợp nhiều loss term để handle các vấn đề đặc thù của CSI sensing:
    1. Coordinate regression loss  — L1/Smooth-L1 cho joint positions
    2. Visibility loss             — BCE với logits cho occlusion/out-of-range
    3. Bone length consistency     — skeleton phải có tỷ lệ cơ thể hợp lý
    4. Temporal smoothness         — pose không được nhảy đột ngột frame-to-frame
    5. Symmetry loss (optional)    — left/right body symmetry
    6. Action classification loss  — cho CLS token (future)

COCO 17 keypoints (dùng làm default):
    0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear
    5: left shoulder, 6: right shoulder, 7: left elbow, 8: right elbow
    9: left wrist, 10: right wrist, 11: left hip, 12: right hip
    13: left knee, 14: right knee, 15: left ankle, 16: right ankle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------
COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COCO_SYMMETRIC_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
]


@dataclass
class LossConfig:
    # Loss weights
    lambda_coord:    float = 1.0
    lambda_vis:      float = 0.5
    lambda_bone:     float = 0.3
    lambda_temporal: float = 0.2
    lambda_symmetry: float = 0.1
    lambda_action:   float = 0.0   # non-zero khi có action labels

    coord_loss_type: str  = "smooth_l1"
    smooth_l1_beta:  float = 1.0
    bone_length_tolerance: float = 0.15
    temporal_smooth_order: int = 1
    vis_pos_weight: float = 2.0

    bones: list = field(default_factory=lambda: COCO_BONES)
    symmetric_pairs: list = field(default_factory=lambda: COCO_SYMMETRIC_PAIRS)


# ===========================================================================
# Individual loss components
# ===========================================================================

def coordinate_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    vis: torch.Tensor,
    loss_type: str = "smooth_l1",
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Regression loss cho joint coordinates, chỉ tính trên visible joints.

    pred/gt:  (B, T, J, C)
    vis:      (B, T, J) — visibility mask (0/1 hoặc float)
    """
    vis_mask = vis.unsqueeze(-1).expand_as(pred)  # (B, T, J, C)

    pred_vis = pred * vis_mask
    gt_vis   = gt   * vis_mask

    if loss_type == "l1":
        loss = F.l1_loss(pred_vis, gt_vis, reduction="sum")
    elif loss_type == "l2":
        loss = F.mse_loss(pred_vis, gt_vis, reduction="sum")
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred_vis, gt_vis, reduction="sum", beta=beta)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    n_visible = vis_mask.sum().clamp(min=1.0)
    return loss / n_visible


def visibility_loss(
    pred_vis_logits: torch.Tensor,
    gt_vis: torch.Tensor,
    pos_weight: float = 2.0,
) -> torch.Tensor:
    """
    Binary cross-entropy với LOGITS cho joint visibility prediction.
    """
    loss = F.binary_cross_entropy_with_logits(
        pred_vis_logits, gt_vis, pos_weight=pos_weight
    )
    return loss


def bone_length_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    bones: list[tuple[int, int]],
    vis: torch.Tensor | None = None,
    tolerance: float = 0.15,
) -> torch.Tensor:
    """
    Penalize khi predicted bone length khác GT quá tolerance (default 15%).
    Chỉ tính khi cả 2 joints visible.
    """
    bone_losses = []

    for parent, child in bones:
        pred_bone = pred[..., parent, :] - pred[..., child, :]
        pred_len  = pred_bone.norm(dim=-1)

        gt_bone = gt[..., parent, :] - gt[..., child, :]
        gt_len  = gt_bone.norm(dim=-1).detach()

        if vis is not None:
            mask = (vis[..., parent] * vis[..., child]).float()
        else:
            mask = torch.ones_like(pred_len)

        rel_error = (pred_len - gt_len).abs() / (gt_len + 1e-6)
        bone_loss = F.relu(rel_error - tolerance)
        bone_loss = (bone_loss * mask).sum() / mask.sum().clamp(min=1.0)
        bone_losses.append(bone_loss)

    return torch.stack(bone_losses).mean()


def temporal_smoothness_loss(
    pred: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    Penalize sudden jumps trong predicted pose sequence.
    order=1: velocity smooth, order=2: acceleration smooth
    """
    if order == 1:
        diff = pred[:, 1:] - pred[:, :-1]
        loss = diff.abs().mean()
    elif order == 2:
        diff = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        loss = diff.abs().mean()
    else:
        raise ValueError(f"order phải là 1 hoặc 2, got {order}")
    return loss


def symmetry_loss(
    pred: torch.Tensor,
    symmetric_pairs: list[tuple[int, int]],
    vis: torch.Tensor | None = None,
) -> torch.Tensor:
    """Enforce left-right body symmetry."""
    losses = []
    for left, right in symmetric_pairs:
        left_coord  = pred[..., left,  :]
        right_coord = pred[..., right, :]
        diff = (left_coord - right_coord).abs().mean(dim=-1)

        if vis is not None:
            mask = (vis[..., left] * vis[..., right]).float()
            loss = (diff * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = diff.mean()
        losses.append(loss)
    return torch.stack(losses).mean()


def action_classification_loss(
    action_logits: torch.Tensor,
    action_labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss cho action recognition từ CLS token."""
    return F.cross_entropy(action_logits, action_labels)


# ===========================================================================
# Main loss class
# ===========================================================================
class RFPoseLoss(nn.Module):
    """
    Tổng hợp tất cả loss terms cho RF-WorldPose training.

    Usage:
        criterion = RFPoseLoss(LossConfig())
        loss, breakdown = criterion(pred_dict, gt_dict, action_labels=...)
        loss.backward()

    Expected pred_dict:
        'coords':     (B, T, J, C)  — predicted joint coordinates
        'vis_logits': (B, T, J)     — visibility RAW LOGITS (không sigmoid)
        'action_logits': (B, num_actions) — optional, từ CLS token

    Expected gt_dict:
        'coords': (B, T, J, C)  — ground truth coordinates
        'vis':    (B, T, J)     — ground truth visibility (0/1)
    """

    def __init__(self, config: LossConfig | None = None):
        super().__init__()
        self.cfg = config or LossConfig()

    def forward(
        self,
        pred_dict: dict[str, torch.Tensor],
        gt_dict:   dict[str, torch.Tensor],
        action_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        pred_coords = pred_dict["coords"]
        pred_vis_logits = pred_dict["vis_logits"]  # raw logits
        gt_coords   = gt_dict["coords"]
        gt_vis      = gt_dict["vis"]

        breakdown = {}

        # 1. Coordinate loss
        l_coord = coordinate_loss(
            pred_coords, gt_coords, gt_vis,
            loss_type=self.cfg.coord_loss_type,
            beta=self.cfg.smooth_l1_beta,
        )
        breakdown["loss_coord"] = l_coord.item()

        # 2. Visibility loss — dùng BCEWithLogitsLoss với raw logits
        l_vis = visibility_loss(pred_vis_logits, gt_vis, self.cfg.vis_pos_weight)
        breakdown["loss_vis"] = l_vis.item()

        # 3. Bone length consistency
        l_bone = bone_length_loss(
            pred_coords, gt_coords,
            bones=self.cfg.bones, vis=gt_vis,
            tolerance=self.cfg.bone_length_tolerance,
        )
        breakdown["loss_bone"] = l_bone.item()

        # 4. Temporal smoothness
        l_temporal = temporal_smoothness_loss(
            pred_coords, order=self.cfg.temporal_smooth_order
        )
        breakdown["loss_temporal"] = l_temporal.item()

        # 5. Symmetry loss
        l_symmetry = symmetry_loss(pred_coords, self.cfg.symmetric_pairs, gt_vis)
        breakdown["loss_symmetry"] = l_symmetry.item()

        # 6. Action classification (optional)
        l_action = torch.tensor(0.0, device=pred_coords.device)
        if action_labels is not None and "action_logits" in pred_dict:
            l_action = action_classification_loss(
                pred_dict["action_logits"], action_labels
            )
            breakdown["loss_action"] = l_action.item()

        # Total loss
        total = (
            self.cfg.lambda_coord    * l_coord +
            self.cfg.lambda_vis      * l_vis +
            self.cfg.lambda_bone     * l_bone +
            self.cfg.lambda_temporal * l_temporal +
            self.cfg.lambda_symmetry * l_symmetry +
            self.cfg.lambda_action   * l_action
        )
        breakdown["loss_total"] = total.item()

        return total, breakdown


# ===========================================================================
# Evaluation metrics
# ===========================================================================
class MPJPE(nn.Module):
    """Mean Per Joint Position Error."""

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        vis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        error = (pred - gt).norm(dim=-1)  # (B, T, J)
        if vis is not None:
            error = error * vis
            return error.sum() / vis.sum().clamp(min=1.0)
        return error.mean()


class PA_MPJPE(nn.Module):
    """
    Procrustes-Aligned MPJPE — vectorized hoàn toàn.
    Loại bỏ global rotation/translation/scale errors.
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        B, T, J, C = pred.shape
        pred_flat = pred.reshape(B * T, J, C)
        gt_flat   = gt.reshape(B * T, J, C)
        aligned = self._procrustes_batch(pred_flat, gt_flat)
        error = (aligned - gt_flat).norm(dim=-1).mean()
        return error

    @staticmethod
    def _procrustes_batch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        gt_c   = gt   - gt.mean(dim=1, keepdim=True)

        pred_norm = pred_c.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        gt_norm   = gt_c.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        pred_s = pred_c / pred_norm
        gt_s   = gt_c / gt_norm

        M = gt_s.transpose(1, 2) @ pred_s
        U, _, Vh = torch.linalg.svd(M)

        det_sign = torch.linalg.det(U @ Vh).sign().unsqueeze(-1).unsqueeze(-1)
        C_dim = pred.shape[-1]
        S_diag = torch.ones(pred.shape[0], C_dim, device=pred.device)
        S_diag[:, -1] = det_sign.squeeze()
        R = U @ (S_diag.unsqueeze(1) * Vh)

        scale = gt_norm / pred_norm
        aligned = scale * (pred_c @ R.transpose(1, 2))
        aligned = aligned + gt.mean(dim=1, keepdim=True)
        return aligned


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, J, C = 4, 100, 17, 3

    pred_dict = {
        "coords": torch.randn(B, T, J, C),
        "vis_logits": torch.randn(B, T, J),  # raw logits
    }
    gt_dict = {
        "coords": torch.randn(B, T, J, C),
        "vis": (torch.rand(B, T, J) > 0.3).float(),
    }

    cfg = LossConfig(lambda_coord=1.0, lambda_vis=0.5, lambda_bone=0.3)
    loss_fn = RFPoseLoss(cfg)
    total, breakdown = loss_fn(pred_dict, gt_dict)

    print(f"Total loss: {total.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k:20s}: {v:.4f}")

    mpjpe = MPJPE()(pred_dict["coords"], gt_dict["coords"], gt_dict["vis"])
    print(f"\nMPJPE: {mpjpe.item():.4f}")
