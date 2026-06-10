"""
loss_aug.py
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

RFPOSE_13_BONES = [
    (0, 1), (0, 2),
    (1, 3), (3, 5), (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (9, 11), (8, 10), (10, 12),
]
RFPOSE_13_SYMMETRIC = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


def skeleton_for_joints(n_joints: int) -> tuple[list, list]:
    if n_joints == 13:
        return RFPOSE_13_BONES, RFPOSE_13_SYMMETRIC
    if n_joints == 17:
        return COCO_BONES, COCO_SYMMETRIC_PAIRS
    raise ValueError(f"No skeleton definition for n_joints={n_joints}")


@dataclass
class LossConfig:
    lambda_coord:    float = 1.0
    lambda_vis:      float = 0.5
    lambda_bone:     float = 0.3
    lambda_temporal: float = 0.2
    lambda_symmetry: float = 0.1
    lambda_action:   float = 0.0

    weighting_mode: str = "static"
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
    pred: torch.Tensor, gt: torch.Tensor, vis: torch.Tensor, 
    loss_type: str = "smooth_l1", beta: float = 1.0
) -> torch.Tensor:
    vis_mask = vis.unsqueeze(-1).expand_as(pred)
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
    pred_vis_logits: torch.Tensor, gt_vis: torch.Tensor, pos_weight: float = 2.0
) -> torch.Tensor:
    pw = torch.tensor(pos_weight, device=pred_vis_logits.device)
    return F.binary_cross_entropy_with_logits(pred_vis_logits, gt_vis, pos_weight=pw)


def bone_length_loss(
    pred: torch.Tensor, gt: torch.Tensor, bones: list[tuple[int, int]],
    vis: torch.Tensor | None = None, tolerance: float = 0.15
) -> torch.Tensor:
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


def temporal_smoothness_loss(pred: torch.Tensor, order: int = 1) -> torch.Tensor:
    if order == 1:
        diff = pred[:, 1:] - pred[:, :-1]
        return diff.abs().mean()
    elif order == 2:
        diff = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        return diff.abs().mean()
    raise ValueError(f"order phải là 1 hoặc 2, got {order}")


def symmetry_loss(
    pred: torch.Tensor, symmetric_pairs: list[tuple[int, int]], vis: torch.Tensor | None = None
) -> torch.Tensor:
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
    action_logits: torch.Tensor, action_labels: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(action_logits, action_labels)


# ===========================================================================
# Main loss class
# ===========================================================================
class RFPoseLoss(nn.Module):
    TASK_NAMES = ["coord", "vis", "bone", "temporal", "symmetry", "action"]

    def __init__(self, config: LossConfig | None = None):
        super().__init__()
        self.cfg = config or LossConfig()
        self.mode = self.cfg.weighting_mode

        if self.mode == "uncertainty":
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1)) for name in self.TASK_NAMES
            })

    def _uncertainty_weight(self, loss: torch.Tensor, task_name: str) -> torch.Tensor:
        s = self.log_vars[task_name]
        return 0.5 * torch.exp(-s) * loss + 0.5 * s

    def _static_weight(self, loss: torch.Tensor, task_name: str) -> torch.Tensor:
        w = {
            "coord": self.cfg.lambda_coord, "vis": self.cfg.lambda_vis,
            "bone": self.cfg.lambda_bone, "temporal": self.cfg.lambda_temporal,
            "symmetry": self.cfg.lambda_symmetry, "action": self.cfg.lambda_action,
        }
        return w[task_name] * loss

    def forward(
        self,
        pred_dict: dict[str, torch.Tensor],
        gt_dict:   dict[str, torch.Tensor],
        action_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        breakdown = {}
        device = pred_dict.get("coords", pred_dict.get("action_logits")).device
        
        has_pose = "coords" in gt_dict and "vis" in gt_dict
        has_action = action_labels is not None and "action_logits" in pred_dict

        l_coord = l_vis = l_bone = l_temporal = l_symmetry = l_action = torch.tensor(0.0, device=device)

        # 1. Tính Loss Pose an toàn
        if has_pose:
            pred_coords = pred_dict["coords"]
            pred_vis_logits = pred_dict["vis_logits"]
            gt_coords = gt_dict["coords"]
            gt_vis = gt_dict["vis"]

            l_coord = coordinate_loss(pred_coords, gt_coords, gt_vis, loss_type=self.cfg.coord_loss_type, beta=self.cfg.smooth_l1_beta)
            l_vis = visibility_loss(pred_vis_logits, gt_vis, self.cfg.vis_pos_weight)
            l_bone = bone_length_loss(pred_coords, gt_coords, bones=self.cfg.bones, vis=gt_vis, tolerance=self.cfg.bone_length_tolerance)
            l_temporal = temporal_smoothness_loss(pred_coords, order=self.cfg.temporal_smooth_order)
            l_symmetry = symmetry_loss(pred_coords, self.cfg.symmetric_pairs, gt_vis)

            breakdown["loss_coord"] = l_coord.item()
            breakdown["loss_vis"] = l_vis.item()
            breakdown["loss_bone"] = l_bone.item()
            breakdown["loss_temporal"] = l_temporal.item()
            breakdown["loss_symmetry"] = l_symmetry.item()

        # 2. Tính Loss Action an toàn
        if has_action:
            l_action = action_classification_loss(pred_dict["action_logits"], action_labels)
            breakdown["loss_action"] = l_action.item()

        # 3. Trả về 0 nếu không có gì để học
        if not has_pose and not has_action:
            breakdown["loss_total"] = 0.0
            return torch.tensor(0.0, requires_grad=True, device=device), breakdown

        # 4. Gom Loss
        losses = {
            "coord": l_coord, "vis": l_vis, "bone": l_bone,
            "temporal": l_temporal, "symmetry": l_symmetry, "action": l_action,
        }
        pose_keys = ["coord", "vis", "bone", "temporal", "symmetry"]

        total = torch.tensor(0.0, device=device)
        
        for name, loss in losses.items():
            if not has_pose and name in pose_keys:
                continue
            if not has_action and name == "action":
                continue

            if self.mode == "uncertainty":
                weighted = self._uncertainty_weight(loss, name)
                total = total + weighted
                
                s = self.log_vars[name].item()
                breakdown[f"w_{name}"] = float(torch.exp(torch.tensor(-s)).item())
                breakdown[f"s_{name}"] = s
            else:
                total = total + self._static_weight(loss, name)

        breakdown["loss_total"] = total.item()
        return total, breakdown


# ===========================================================================
# Evaluation metrics
# ===========================================================================
class MPJPE(nn.Module):
    """Mean Per Joint Position Error."""
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, vis: torch.Tensor | None = None) -> torch.Tensor:
        error = (pred - gt).norm(dim=-1)
        if vis is not None:
            error = error * vis
            return error.sum() / vis.sum().clamp(min=1.0)
        return error.mean()


class PA_MPJPE(nn.Module):
    """Procrustes-Aligned MPJPE — vectorized hoàn toàn."""
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        B, T, J, C = pred.shape
        pred_flat = pred.reshape(B * T, J, C)
        gt_flat   = gt.reshape(B * T, J, C)
        aligned = self._procrustes_batch(pred_flat, gt_flat)
        return (aligned - gt_flat).norm(dim=-1).mean()

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
        return aligned + gt.mean(dim=1, keepdim=True)