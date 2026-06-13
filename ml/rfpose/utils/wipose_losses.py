"""WiPose loss functions (paper Section 3.3).

J = Lp + β·Ls + γ·Lr

    Lp = position loss (L2 norm)
    Ls = smooth loss (Huber norm on velocity difference)
    Lr = rotation loss (Huber norm on relative position error)

Reference: Jiang et al., MobiCom 2020, Eq. 2-7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfpose.models.wipose_net import SKELETON_CONFIGS


class WiPoseLoss(nn.Module):
    """Combined WiPose loss: Lp + β·Ls + γ·Lr."""

    def __init__(self, beta: float = 1.0, gamma: float = 1.0, n_joints: int = 18):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

        _, parent_map, _ = SKELETON_CONFIGS[n_joints]
        children = []
        parent_list = []
        for child in sorted(parent_map.keys()):
            children.append(child)
            parent_list.append(parent_map[child])
        self.register_buffer("child_idx", torch.tensor(children, dtype=torch.long))
        self.register_buffer("parent_idx", torch.tensor(parent_list, dtype=torch.long))

    def position_loss(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        """Lp = (1/T) Σ_t (1/N) Σ_i ||p̂ - p||_2  (Eq. 2)."""
        return (pred - gt).norm(dim=-1).mean()

    def smooth_loss(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        """Ls: Huber norm on velocity difference (Eq. 3).

        Forces pred velocity to match GT velocity for temporal smoothness.
        """
        pred_vel = pred[:, 1:] - pred[:, :-1]  # (B, T-1, N, 3)
        gt_vel = gt[:, 1:] - gt[:, :-1]

        diff = pred_vel - gt_vel  # (B, T-1, N, 3)
        return F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="mean")

    def rotation_loss(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        """Lr: Huber norm on relative position error (Eq. 6).

        Penalizes error in bone vectors (child - parent).
        """
        pred_rel = pred[:, :, self.child_idx] - pred[:, :, self.parent_idx]
        gt_rel = gt[:, :, self.child_idx] - gt[:, :, self.parent_idx]

        diff = pred_rel - gt_rel
        return F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="mean")

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred: (B, T, N, 3) predicted joint positions
            gt:   (B, T, N, 3) ground truth joint positions
        Returns:
            total_loss, breakdown dict
        """
        lp = self.position_loss(pred, gt)
        ls = self.smooth_loss(pred, gt)
        lr = self.rotation_loss(pred, gt)

        total = lp + self.beta * ls + self.gamma * lr

        breakdown = {
            "loss_total": total.item(),
            "loss_position": lp.item(),
            "loss_smooth": ls.item(),
            "loss_rotation": lr.item(),
        }
        return total, breakdown
