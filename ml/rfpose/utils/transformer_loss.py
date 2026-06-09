import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMaskedMPJPELoss(nn.Module):
    """
    MPJPE trên (B, T, J, 3) với:
        pose_mask:   (B, J) hoặc (B, T, J) — binary, bỏ joints không có label
        vis_logits:  (B, T, J) raw logits — dùng sigmoid làm confidence weight

    Công thức:
        conf  = sigmoid(vis_logits).detach()      # (B, T, J), stop-gradient
        weight = pose_mask * conf                 # chỉ weight joints hợp lệ
        loss   = sum(||pred - target|| * weight) / sum(weight)

    Lý do weight theo vis_confidence:
        - Joints model tự tin visible → loss cao hơn → gradient mạnh hơn
        - Joints bị occlude (conf thấp) → đóng góp ít → tránh noisy gradient
        - detach() để vis_logits chỉ ảnh hưởng qua VisibilityLoss, không loop ngược
    """
    def forward(
        self,
        pred: torch.Tensor,        # (B, T, J, 3) hoặc (B, J, 3)
        target: torch.Tensor,      # cùng shape với pred
        pose_mask: torch.Tensor,   # (B, J) hoặc (B, T, J)
        vis_logits: torch.Tensor,  # (B, T, J) raw logits từ model
    ) -> torch.Tensor:

        # Normalise target lên (B, T, J, 3) nếu GT chỉ có 1 frame per sample
        # pred luôn là (B, T, J, 3) từ V3 decoder
        if target.ndim == 3:
            # (B, J, 3) → expand thành (B, T, J, 3) để broadcast đúng
            T = pred.shape[1]
            target = target.unsqueeze(1).expand(-1, T, -1, -1)

        error = torch.norm(pred - target, dim=-1)   # (B, T, J)

        # Broadcast pose_mask (B, J) → (B, T, J)
        if pose_mask.ndim == 2:
            pose_mask = pose_mask.unsqueeze(1).expand_as(error)

        # Visibility confidence — stop-gradient để không tạo feedback loop
        conf   = torch.sigmoid(vis_logits).detach()     # (B, T, J)
        weight = pose_mask * conf                        # (B, T, J)

        n_valid = weight.sum().clamp(min=1.0)
        return (error * weight).sum() / n_valid


class VisibilityLoss(nn.Module):
    """
    BCE loss giữa vis_logits và vis_target.

    vis_target = pose_mask broadcast lên (B, T, J):
        - pose_mask = 1 → joint có label → target visibility = 1 (visible)
        - pose_mask = 0 → joint bị occlude → target visibility = 0

    Chỉ tính loss trên joints có pose_mask = 1 để tránh supervise
    joints hoàn toàn không có ground truth.
    """
    def forward(
        self,
        vis_logits: torch.Tensor,  # (B, T, J) raw logits
        pose_mask: torch.Tensor,   # (B, J) hoặc (B, T, J) binary
    ) -> torch.Tensor:
        T = vis_logits.shape[1]

        # Broadcast pose_mask lên (B, T, J)
        if pose_mask.ndim == 2:
            vis_target = pose_mask.unsqueeze(1).expand_as(vis_logits)   # (B, T, J)
            mask_2d    = pose_mask.unsqueeze(1).expand_as(vis_logits)
        else:
            vis_target = pose_mask.float()
            mask_2d    = pose_mask

        bce     = F.binary_cross_entropy_with_logits(
            vis_logits, vis_target.float(), reduction="none"
        )
        # Chỉ tính trên joints có label gốc (pose_mask=1)
        masked  = bce * mask_2d
        n_valid = mask_2d.sum().clamp(min=1.0)
        return masked.sum() / n_valid


class ActionLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits, action_label, action_mask):
        per_sample = self.ce(logits, action_label)
        return (per_sample * action_mask).sum() / action_mask.sum().clamp(min=1.0)


class PresenceLoss(nn.Module):
    """BCE loss cho presence_logit (B,) — người có mặt hay không."""
    def forward(self, presence_logit, presence_target):
        return F.binary_cross_entropy_with_logits(presence_logit, presence_target)


class KendallLoss(nn.Module):
    """Auto-balance N losses bằng uncertainty weighting."""
    def __init__(self, n_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        assert len(losses) == self.log_vars.shape[0]
        total = sum(
            torch.exp(-lv) * l + lv
            for lv, l in zip(self.log_vars, losses)
        )
        info = {f"loss_{i}": l.item() for i, l in enumerate(losses)}
        info["loss_total"] = total.item()
        return total, info


class RFPoseLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.mpjpe    = TemporalMaskedMPJPELoss()
        self.vis_loss = VisibilityLoss()
        self.act_loss = ActionLoss(label_smoothing)
        self.pres_loss = PresenceLoss()
        self.combiner  = KendallLoss(n_tasks=3)

    def forward(self, pred_coords, pred_vis, pred_action, pred_presence,
                target_joints, pose_mask, action_label, action_mask,
                presence_target=None):

        # MPJPE weighted by vis_confidence (stop-grad)
        l_pose   = self.mpjpe(pred_coords, target_joints, pose_mask, pred_vis)
        l_action = self.act_loss(pred_action, action_label, action_mask)
        # VisibilityLoss: vis_target được suy ra từ pose_mask bên trong
        l_vis    = self.vis_loss(pred_vis, pose_mask)

        # Presence loss (nếu có label)
        l_pres = torch.tensor(0.0, device=pred_coords.device)
        if presence_target is not None:
            l_pres = self.pres_loss(pred_presence, presence_target.float())

        total, info = self.combiner([l_pose, l_action, l_vis])
        info.update({
            "loss_pose":     l_pose.item(),
            "loss_action":   l_action.item(),
            "loss_vis":      l_vis.item(),
            "loss_presence": l_pres.item(),
        })
        return total + l_pres * 0.1, info