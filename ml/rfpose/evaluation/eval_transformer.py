import torch
import numpy as np


def procrustes_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align pred → target theo Procrustes (SVD).
    pred, target: (J, 3)  →  pred_aligned: (J, 3)
    """
    mu_p = pred.mean(0);   mu_t = target.mean(0)
    pc   = pred - mu_p;    tc   = target - mu_t

    norm_p = np.sqrt((pc**2).sum()) + 1e-8
    norm_t = np.sqrt((tc**2).sum()) + 1e-8

    H = (pc / norm_p).T @ (tc / norm_t)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T

    return (norm_t / norm_p) * (pc @ R.T) + mu_t


def _pa_mpjpe_samples(pred_np, target_np, mask_np):
    """
    pred_np, target_np: (N, J, 3)
    mask_np:            (N, J)
    returns list of per-sample PA-MPJPE
    """
    errors = []
    for i in range(len(pred_np)):
        valid = mask_np[i] > 0.5
        if valid.sum() < 3:
            continue
        p_aligned = procrustes_align(pred_np[i][valid], target_np[i][valid])
        errors.append(np.linalg.norm(p_aligned - target_np[i][valid], axis=-1).mean())
    return errors


# ──────────────────────────────────────────────
# PoseEvaluator
# ──────────────────────────────────────────────

class PoseEvaluator:
    """
    Accumulate predictions qua nhiều batch, tính metrics cuối epoch.

    Xử lý output:
        coords:         (B, T, J, 3)  → mean pool T → (B, J, 3) trước khi eval
        vis_logits:     (B, T, J)     → mean pool T → (B, J), threshold 0.5
        action_logits:  (B, num_actions)
        presence_logit: (B,)
    """

    def __init__(self, num_actions: int = 28):
        self.num_actions = num_actions
        self.reset()

    def reset(self):
        self._mpjpe:       list[float] = []
        self._pa_mpjpe:    list[float] = []
        self._pred_action: list[int]   = []
        self._true_action: list[int]   = []
        self._vis_correct: list[float] = []   # fraction correct per sample
        self._pres_correct: list[int]  = []   # 0/1 per sample (nếu có label)

    @torch.no_grad()
    def update(
        self,
        model_out:      dict,               # output dict từ RFPoseModel.forward()
        target_joints:  torch.Tensor,       # (B, J, 3)  — GT skeleton
        pose_mask:      torch.Tensor,       # (B, J)     — 1=valid joint
        action_label:   torch.Tensor,       # (B,)  long
        action_mask:    torch.Tensor,       # (B,)  float
        presence_target: torch.Tensor | None = None,  # (B,) binary float, optional
    ):
        coords        = model_out["coords"].float()          # (B, T, J, 3)
        vis_logits    = model_out["vis_logits"].float()      # (B, T, J)
        action_logits = model_out["action_logits"].float()   # (B, num_actions)
        presence_logit= model_out["presence_logit"].float()  # (B,)

        B = coords.shape[0]

        # ── Mean pool T → (B, J, 3) / (B, J) ──
        pred_joints_avg = coords.mean(dim=1)          # (B, J, 3)
        vis_avg         = vis_logits.mean(dim=1)      # (B, J)  — averaged logits

        # ── MPJPE ──
        error = torch.norm(pred_joints_avg - target_joints, dim=-1)  # (B, J)
        for i in range(B):
            valid = pose_mask[i] > 0.5
            if valid.sum() == 0:
                continue
            self._mpjpe.append(error[i][valid].mean().item())

        # ── PA-MPJPE ──
        pa_errs = _pa_mpjpe_samples(
            pred_joints_avg.cpu().numpy(),
            target_joints.cpu().numpy(),
            pose_mask.cpu().numpy(),
        )
        self._pa_mpjpe.extend(pa_errs)

        # ── Action ──
        pred_cls = action_logits.argmax(dim=-1).cpu().tolist()
        true_cls = action_label.cpu().tolist()
        for p, t, m in zip(pred_cls, true_cls, action_mask.cpu().tolist()):
            if m > 0.5:
                self._pred_action.append(p)
                self._true_action.append(t)

        # ── Visibility accuracy ──
        # GT visibility = pose_mask (1=visible), pred = sigmoid(vis_avg) > 0.5
        vis_pred = (torch.sigmoid(vis_avg) > 0.5).float()   # (B, J)
        vis_gt   = (pose_mask > 0.5).float()                 # (B, J)
        for i in range(B):
            # Tính trên tất cả joints (cả visible lẫn occluded)
            acc_i = (vis_pred[i] == vis_gt[i]).float().mean().item()
            self._vis_correct.append(acc_i)

        # ── Presence accuracy (nếu có label) ──
        if presence_target is not None:
            pres_pred = (torch.sigmoid(presence_logit) > 0.5).long()
            pres_gt   = (presence_target > 0.5).long()
            self._pres_correct.extend((pres_pred == pres_gt).cpu().tolist())

    def compute(self) -> dict:
        """Tính tất cả metrics, trả về dict."""
        nan = float("nan")
        results = {
            "mpjpe":        float(np.mean(self._mpjpe))       if self._mpjpe    else nan,
            "pa_mpjpe":     float(np.mean(self._pa_mpjpe))    if self._pa_mpjpe else nan,
            "vis_acc":      float(np.mean(self._vis_correct)) if self._vis_correct else nan,
            "presence_acc": float(np.mean(self._pres_correct)) if self._pres_correct else nan,
        }

        if self._pred_action:
            pred_arr = np.array(self._pred_action)
            true_arr = np.array(self._true_action)
            results["action_acc"] = float((pred_arr == true_arr).mean())
            results["macro_f1"]   = _macro_f1(pred_arr, true_arr, self.num_actions)
        else:
            results["action_acc"] = nan
            results["macro_f1"]   = nan

        return results

    def log_str(self) -> str:
        r = self.compute()
        def fmt(v): return f"{v:.1f}" if not np.isnan(v) else "N/A"
        def fmtp(v): return f"{v:.3f}" if not np.isnan(v) else "N/A"
        return (
            f"MPJPE={fmt(r['mpjpe'])}  "
            f"PA-MPJPE={fmt(r['pa_mpjpe'])}  "
            f"ActAcc={fmtp(r['action_acc'])}  "
            f"F1={fmtp(r['macro_f1'])}  "
            f"VisAcc={fmtp(r['vis_acc'])}  "
            f"PresAcc={fmtp(r['presence_acc'])}"
        )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _macro_f1(pred: np.ndarray, true: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return float(np.mean(f1s))


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, J = 4, 60, 13

    evaluator = PoseEvaluator(num_actions=28)
    evaluator.reset()

    for _ in range(3):
        fake_out = {
            "coords":          torch.randn(B, T, J, 3),
            "vis_logits":      torch.randn(B, T, J),
            "action_logits":   torch.randn(B, 28),
            "presence_logit":  torch.randn(B),
        }
        evaluator.update(
            model_out       = fake_out,
            target_joints   = torch.randn(B, J, 3),
            pose_mask       = torch.ones(B, J),
            action_label    = torch.randint(0, 28, (B,)),
            action_mask     = torch.ones(B),
            presence_target = torch.ones(B),
        )

    print("Sanity check:", evaluator.log_str())