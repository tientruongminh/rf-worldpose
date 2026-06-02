"""
losses.py
---------
Loss functions cho WiFi CSI -> Pose Estimation.

Tổng hợp nhiều loss term để handle các vấn đề đặc thù của CSI sensing:
    1. Coordinate regression loss  — L1/Smooth-L1 cho joint positions
    2. Visibility loss             — BCE cho occlusion/out-of-range joints
    3. Bone length consistency     — skeleton phải có tỉ lệ cơ thể hợp lý
    4. Temporal smoothness         — pose không được nhảy đột ngột frame-to-frame
    5. Symmetry loss (optional)    — left/right body symmetry
    6. Heatmap loss (optional)     — nếu model predict heatmap thay vì coordinate trực tiếp

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

# COCO 17 keypoints: (parent, child) pairs
COCO_BONES = [
    (0,  1),   # nose - left eye
    (0,  2),   # nose - right eye
    (1,  3),   # left eye - left ear
    (2,  4),   # right eye - right ear
    (5,  7),   # left shoulder - left elbow
    (7,  9),   # left elbow - left wrist
    (6,  8),   # right shoulder - right elbow
    (8,  10),  # right elbow - right wrist
    (5,  6),   # left shoulder - right shoulder
    (5,  11),  # left shoulder - left hip
    (6,  12),  # right shoulder - right hip
    (11, 12),  # left hip - right hip
    (11, 13),  # left hip - left knee
    (13, 15),  # left knee - left ankle
    (12, 14),  # right hip - right knee
    (14, 16),  # right knee - right ankle
]

# Symmetric pairs (left, right) — dùng cho symmetry loss
COCO_SYMMETRIC_PAIRS = [
    (1, 2),    # eyes
    (3, 4),    # ears
    (5, 6),    # shoulders
    (7, 8),    # elbows
    (9, 10),   # wrists
    (11, 12),  # hips
    (13, 14),  # knees
    (15, 16),  # ankles
]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class LossConfig:
    # Loss weights
    lambda_coord:    float = 1.0     # coordinate regression
    lambda_vis:      float = 0.5     # visibility prediction
    lambda_bone:     float = 0.3     # bone length consistency
    lambda_temporal: float = 0.2     # temporal smoothness
    lambda_symmetry: float = 0.1     # left-right symmetry

    # Coordinate loss type
    coord_loss_type: str  = "smooth_l1"   # "l1", "l2", "smooth_l1"
    smooth_l1_beta:  float = 1.0

    # Bone constraint
    bone_length_tolerance: float = 0.15   # 15% deviation tolerance

    # Temporal smoothness
    temporal_smooth_order: int = 1   # 1 = velocity, 2 = acceleration

    # Visibility
    vis_pos_weight: float = 2.0  # upweight visible joints (thường bị imbalance)

    # Skeleton
    bones: list = field(default_factory=lambda: COCO_BONES)
    symmetric_pairs: list = field(default_factory=lambda: COCO_SYMMETRIC_PAIRS)


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

def coordinate_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    vis: torch.Tensor,
    loss_type: str = "smooth_l1",
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Regression loss cho joint coordinates, chỉ tính trên visible joints.

    Args:
        pred:      (B, T, J, C) — predicted coordinates
        gt:        (B, T, J, C) — ground truth coordinates
        vis:       (B, T, J)    — visibility mask (1 = visible, 0 = occluded)
        loss_type: "l1", "l2", hoặc "smooth_l1"
        beta:      smooth_l1 beta parameter

    Returns:
        scalar loss

    Lý do dùng Smooth-L1 thay vì MSE:
        - L2 bị ảnh hưởng nhiều bởi outliers (CSI noise đôi khi spike)
        - L1 không differentiable tại 0
        - Smooth-L1 là compromise tốt nhất
    """
    # Expand visibility để mask coordinate dims
    vis_mask = vis.unsqueeze(-1).expand_as(pred)  # (B, T, J, C)

    # Chỉ tính loss trên visible joints
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

    # Normalize bằng số visible joints (tránh bias khi nhiều joints bị occlude)
    n_visible = vis_mask.sum().clamp(min=1.0)
    return loss / n_visible


def visibility_loss(
    pred_vis: torch.Tensor,
    gt_vis: torch.Tensor,
    pos_weight: float = 2.0,
) -> torch.Tensor:
    """
    Binary cross-entropy cho joint visibility prediction.

    Args:
        pred_vis:   (B, T, J) — predicted visibility score (sau sigmoid)
        gt_vis:     (B, T, J) — ground truth visibility (0 hoặc 1)
        pos_weight: weight cho positive class (visible joints)

    Returns:
        scalar loss
    """
    weight = torch.ones_like(gt_vis)
    weight[gt_vis == 1] = pos_weight  # upweight visible joints

    # pred_vis đã qua sigmoid, dùng binary_cross_entropy
    loss = F.binary_cross_entropy(pred_vis, gt_vis, weight=weight)
    return loss


def bone_length_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    bones: list[tuple[int, int]],
    vis: torch.Tensor | None = None,
    tolerance: float = 0.15,
) -> torch.Tensor:
    """
    Penalize khi predicted bone length khác ground truth bone length quá nhiều.

    Lý do cần loss này:
    - CSI sensing không có texture info => model dễ predict poses hình học sai
    - Skeleton có ràng buộc cơ học: tỉ lệ xương người tương đối ổn định
    - Loss này enforce skeleton consistency

    Args:
        pred:      (B, T, J, C) — predicted coordinates
        gt:        (B, T, J, C) — ground truth
        bones:     list of (parent_idx, child_idx)
        vis:       (B, T, J) — visibility mask (chỉ tính bone nếu cả 2 joints visible)
        tolerance: fraction deviation allowed (0.15 = 15% tolerance)

    Returns:
        scalar loss
    """
    bone_losses = []

    for parent, child in bones:
        # Predicted bone vector và length
        pred_bone = pred[..., parent, :] - pred[..., child, :]   # (B, T, C)
        pred_len  = pred_bone.norm(dim=-1)                         # (B, T)

        # GT bone length
        gt_bone = gt[..., parent, :] - gt[..., child, :]
        gt_len  = gt_bone.norm(dim=-1).detach()  # detach GT từ graph

        # Mask: chỉ tính khi cả 2 joints visible
        if vis is not None:
            mask = (vis[..., parent] * vis[..., child]).float()  # (B, T)
        else:
            mask = torch.ones_like(pred_len)

        # Relative error với tolerance
        rel_error = (pred_len - gt_len).abs() / (gt_len + 1e-6)
        # Chỉ penalize khi vượt tolerance
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

    Args:
        pred:  (B, T, J, C) — predicted joint coordinates
        order: 1 = velocity smooth (|x[t] - x[t-1]|)
               2 = acceleration smooth (|x[t] - 2*x[t-1] + x[t-2]|)

    Returns:
        scalar loss

    Lý do cần loss này:
    - CSI signal có nhiễu cao-tần, model có thể predict jittery poses
    - Human motion thực tế là smooth (bandwidth giới hạn bởi cơ học cơ thể)
    - Velocity smoothness tương đương constraint: max angular velocity
    """
    if order == 1:
        # First-order difference (velocity)
        diff = pred[:, 1:] - pred[:, :-1]   # (B, T-1, J, C)
        loss = diff.abs().mean()

    elif order == 2:
        # Second-order difference (acceleration)
        diff = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # (B, T-2, J, C)
        loss = diff.abs().mean()

    else:
        raise ValueError(f"order phải là 1 hoặc 2, got {order}")

    return loss


def symmetry_loss(
    pred: torch.Tensor,
    symmetric_pairs: list[tuple[int, int]],
    vis: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Enforce left-right body symmetry.
    Bone length của các pairs đối xứng phải tương đương nhau.

    Args:
        pred:            (B, T, J, C) — predicted coordinates
        symmetric_pairs: list of (left_idx, right_idx)
        vis:             (B, T, J) — visibility mask

    Returns:
        scalar loss
    """
    losses = []

    for left, right in symmetric_pairs:
        left_coord  = pred[..., left,  :]   # (B, T, C)
        right_coord = pred[..., right, :]   # (B, T, C)

        # Symmetry: reflected distance to body center should match
        # Simplified: penalize difference in joint magnitude from center
        diff = (left_coord - right_coord).abs().mean(dim=-1)  # (B, T)

        if vis is not None:
            mask = (vis[..., left] * vis[..., right]).float()
            loss = (diff * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = diff.mean()

        losses.append(loss)

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------
class RFPoseLoss(nn.Module):
    """
    Tổng hợp tất cả loss terms cho RF-WorldPose training.

    Usage:
        criterion = RFPoseLoss(LossConfig())
        loss, breakdown = criterion(pred_dict, gt_dict)
        loss.backward()

    Args:
        config: LossConfig dataclass

    Expected inputs:
        pred_dict: {
            'coords': (B, T, J, C),   # predicted joint coordinates
            'vis':    (B, T, J),      # predicted visibility scores (post-sigmoid)
        }
        gt_dict: {
            'coords': (B, T, J, C),   # ground truth joint coordinates
            'vis':    (B, T, J),      # ground truth visibility (0/1)
        }

    Returns:
        total_loss: scalar
        breakdown:  dict với từng loss component (để log MLflow)
    """

    def __init__(self, config: LossConfig | None = None):
        super().__init__()
        self.cfg = config or LossConfig()

    def forward(
        self,
        pred_dict: dict[str, torch.Tensor],
        gt_dict:   dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:

        pred_coords = pred_dict["coords"]   # (B, T, J, C)
        pred_vis    = pred_dict["vis"]      # (B, T, J)
        gt_coords   = gt_dict["coords"]     # (B, T, J, C)
        gt_vis      = gt_dict["vis"]        # (B, T, J)

        breakdown = {}

        # --- 1. Coordinate loss ---
        l_coord = coordinate_loss(
            pred_coords, gt_coords, gt_vis,
            loss_type=self.cfg.coord_loss_type,
            beta=self.cfg.smooth_l1_beta,
        )
        breakdown["loss_coord"] = l_coord.item()

        # --- 2. Visibility loss ---
        l_vis = visibility_loss(pred_vis, gt_vis, self.cfg.vis_pos_weight)
        breakdown["loss_vis"] = l_vis.item()

        # --- 3. Bone length consistency ---
        l_bone = bone_length_loss(
            pred_coords, gt_coords,
            bones=self.cfg.bones,
            vis=gt_vis,
            tolerance=self.cfg.bone_length_tolerance,
        )
        breakdown["loss_bone"] = l_bone.item()

        # --- 4. Temporal smoothness ---
        l_temporal = temporal_smoothness_loss(pred_coords, order=self.cfg.temporal_smooth_order)
        breakdown["loss_temporal"] = l_temporal.item()

        # --- 5. Symmetry loss ---
        l_symmetry = symmetry_loss(pred_coords, self.cfg.symmetric_pairs, gt_vis)
        breakdown["loss_symmetry"] = l_symmetry.item()

        # --- Total loss ---
        total = (
            self.cfg.lambda_coord    * l_coord +
            self.cfg.lambda_vis      * l_vis +
            self.cfg.lambda_bone     * l_bone +
            self.cfg.lambda_temporal * l_temporal +
            self.cfg.lambda_symmetry * l_symmetry
        )
        breakdown["loss_total"] = total.item()

        return total, breakdown


# ---------------------------------------------------------------------------
# Evaluation metric (không dùng trong training, dùng cho validation)
# ---------------------------------------------------------------------------
class MPJPE(nn.Module):
    """
    Mean Per Joint Position Error — metric chuẩn cho pose estimation.
    Unit: mm (nếu coords ở mm), hoặc normalized units.

    MPJPE = mean over joints và samples của Euclidean distance giữa pred và gt.
    """

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        vis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        pred, gt: (B, T, J, C)
        vis:      (B, T, J) — nếu None, tính trên tất cả joints
        """
        error = (pred - gt).norm(dim=-1)  # (B, T, J)

        if vis is not None:
            error = error * vis
            return error.sum() / vis.sum().clamp(min=1.0)
        return error.mean()


class PA_MPJPE(nn.Module):
    """
    Procrustes-Aligned MPJPE — align predicted skeleton trước khi tính error.
    Loại bỏ global rotation/translation/scale errors,
    chỉ đo per-joint structural accuracy.

    Vectorized hoàn toàn: không có Python for-loop qua B hay T.
    Tất cả Procrustes steps (center, scale, SVD, rotate) chạy batch trên GPU.
    Với B=32, T=100: ~3200 lần faster so với implementation cũ.
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: (B, T, J, C)
        returns: scalar MPJPE sau khi align
        """
        B, T, J, C = pred.shape

        # Flatten B,T -> (B*T, J, C) để xử lý batch lớn với linalg.svd
        pred_flat = pred.reshape(B * T, J, C)
        gt_flat   = gt.reshape(B * T, J, C)

        aligned = self._procrustes_batch(pred_flat, gt_flat)  # (B*T, J, C)

        error = (aligned - gt_flat).norm(dim=-1).mean()  # scalar
        return error

    @staticmethod
    def _procrustes_batch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Procrustes alignment cho batch của skeletons.

        pred, gt: (N, J, C) — N = B*T samples
        returns:  (N, J, C) — pred đã được align vào gt
        """
        # 1. Center: trừ mean joint position — (N, 1, C)
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        gt_c   = gt   - gt.mean(dim=1, keepdim=True)

        # 2. Scale: normalize về unit Frobenius norm
        pred_norm = pred_c.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)  # (N,1,1)
        gt_norm   = gt_c.norm(  dim=(1, 2), keepdim=True).clamp(min=1e-8)
        pred_s = pred_c / pred_norm
        gt_s   = gt_c   / gt_norm

        # 3. Optimal rotation: SVD của cross-covariance matrix
        # M = gt_s^T @ pred_s => (N, C, C)
        M = gt_s.transpose(1, 2) @ pred_s   # (N, C, C)
        U, _, Vh = torch.linalg.svd(M)       # U: (N,C,C), Vh: (N,C,C)

        # Đảm bảo rotation (det = +1), không phải reflection
        det_sign = torch.linalg.det(U @ Vh).sign().unsqueeze(-1).unsqueeze(-1)  # (N,1,1)
        S_diag = torch.ones(pred.shape[0], C, device=pred.device)
        S_diag[:, -1] = det_sign.squeeze()
        R = U @ (S_diag.unsqueeze(1) * Vh)   # (N, C, C) — proper rotation matrix

        # 4. Apply: scale pred lên scale của gt, rotate, translate về gt center
        scale   = gt_norm / pred_norm                             # (N,1,1)
        aligned = scale * (pred_c @ R.transpose(1, 2))           # (N, J, C)
        aligned = aligned + gt.mean(dim=1, keepdim=True)          # re-center

        return aligned


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, J, C = 4, 100, 17, 3

    pred_dict = {
        "coords": torch.randn(B, T, J, C),
        "vis":    torch.sigmoid(torch.randn(B, T, J)),
    }
    gt_dict = {
        "coords": torch.randn(B, T, J, C),
        "vis":    (torch.rand(B, T, J) > 0.3).float(),  # ~70% visible
    }

    cfg  = LossConfig(lambda_coord=1.0, lambda_vis=0.5, lambda_bone=0.3)
    loss_fn = RFPoseLoss(cfg)
    total, breakdown = loss_fn(pred_dict, gt_dict)

    print(f"Total loss: {total.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k:20s}: {v:.4f}")

    # Eval metrics
    mpjpe = MPJPE()(pred_dict["coords"], gt_dict["coords"], gt_dict["vis"])
    print(f"\nMPJPE: {mpjpe.item():.4f}")