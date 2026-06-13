"""
visualize_pose.py — 3D skeleton visualization (Pred vs GT overlay).

Auto-rotates skeleton so head is on top.
Draws clean 3D human stick figure with color-coded body parts.

Usage:
    python -m rfpose.evaluation.visualize_pose \
        --checkpoint checkpoints/rootrel-mmfi-v1/best.pt \
        --output-dir viz_output_mmfi --num-samples 20
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from rfpose.models.csi_tokenizer import CSITokenizer
from rfpose.models.transformer import CSITransformerPose
from rfpose.data.gold_npz_dataset import GoldNpzDataset, ACTION_LABELS, NUM_ACTIONS

try:
    from rfpose.models.transformer_rootrel import CSITransformerPoseRootRel
except ImportError:
    CSITransformerPoseRootRel = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

JOINT_NAMES = [
    "Head", "L.Shoulder", "R.Shoulder", "L.Elbow", "R.Elbow",
    "L.Wrist", "R.Wrist", "L.Hip", "R.Hip", "L.Knee", "R.Knee",
    "L.Ankle", "R.Ankle",
]

BONES = [
    (0, 1), (0, 2),       # head → shoulders
    (1, 2),               # shoulder bar
    (1, 3), (3, 5),       # L arm
    (2, 4), (4, 6),       # R arm
    (1, 7), (2, 8),       # torso sides
    (7, 8),               # hip bar
    (7, 9), (9, 11),      # L leg
    (8, 10), (10, 12),    # R leg
]

# Color per bone group
TORSO_C = "#3D5A80"
L_ARM_C = "#E07A5F"
R_ARM_C = "#F2CC8F"
L_LEG_C = "#81B29A"
R_LEG_C = "#F4A261"
HEAD_C  = "#5B8C5A"

BONE_COLORS = {
    (0, 1): HEAD_C, (0, 2): HEAD_C,
    (1, 2): TORSO_C,
    (1, 7): TORSO_C, (2, 8): TORSO_C, (7, 8): TORSO_C,
    (1, 3): L_ARM_C, (3, 5): L_ARM_C,
    (2, 4): R_ARM_C, (4, 6): R_ARM_C,
    (7, 9): L_LEG_C, (9, 11): L_LEG_C,
    (8, 10): R_LEG_C, (10, 12): R_LEG_C,
}

ZERO_THRESH = 0.005


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    d, m = cfg["data"], cfg["model"]
    variant = m.get("variant", "base")

    tokenizer = CSITokenizer(
        n_subcarriers=d["n_subcarriers"],
        patch_size=m["patch_size"],
        d_model=m["d_model"],
        max_seq_len=d.get("window_size", 60) + 10,
        n_nodes=d.get("n_nodes", 1),
        dropout=m.get("dropout", 0.1),
    ).to(device)

    common = dict(
        n_patches=tokenizer.n_patches,
        d_model=m["d_model"],
        spatial_heads=m["spatial_heads"],
        temporal_heads=m["temporal_heads"],
        n_spatial_layers=m["n_spatial_layers"],
        n_temporal_layers=m["n_temporal_layers"],
        n_decoder_layers=m["n_decoder_layers"],
        n_decoder_temporal_layers=m.get("n_decoder_temporal_layers", 2),
        n_joints=d.get("n_joints", 13),
        predict_3d=m.get("predict_3d", True),
        causal_temporal=m.get("causal_temporal", False),
        dropout=m.get("dropout", 0.1),
        ffn_mult=m.get("ffn_mult", 4),
        n_nodes=d.get("n_nodes", 1),
        num_actions=m.get("num_actions", NUM_ACTIONS),
    )

    if variant == "rootrel" and CSITransformerPoseRootRel is not None:
        model = CSITransformerPoseRootRel(**common).to(device)
    else:
        model = CSITransformerPose(**common).to(device)

    tokenizer.load_state_dict(ckpt["tokenizer"])
    model.load_state_dict(ckpt["model"])
    tokenizer.eval()
    model.eval()
    return tokenizer, model, cfg


def _valid(joints: np.ndarray) -> np.ndarray:
    return np.linalg.norm(joints, axis=-1) > ZERO_THRESH


def _compute_transform(joints_seq: np.ndarray):
    """Compute rotation matrix and center from GT to make skeleton upright.
    Returns (center, R) where R rotates so head is +Z, facing camera."""
    valid = _valid(joints_seq)
    if not valid.any():
        return np.zeros(3), np.eye(3)

    torso_idx = [1, 2, 7, 8]
    torso_mask = valid[:, torso_idx]
    if torso_mask.any():
        torso_pts = joints_seq[:, torso_idx][torso_mask]
        center = torso_pts.mean(axis=0)
    else:
        center = joints_seq[valid].mean(axis=0)

    pts = joints_seq - center

    head_frames = valid[:, 0]
    ankle_valid = valid[:, 11] & valid[:, 12]
    hip_valid = valid[:, 7] & valid[:, 8]

    if head_frames.any() and ankle_valid.any():
        head_mean = pts[head_frames, 0].mean(axis=0)
        ank_mean = (pts[ankle_valid, 11] + pts[ankle_valid, 12]).mean(axis=0) / 2
        up = head_mean - ank_mean
    elif head_frames.any() and hip_valid.any():
        head_mean = pts[head_frames, 0].mean(axis=0)
        hip_mean = (pts[hip_valid, 7] + pts[hip_valid, 8]).mean(axis=0) / 2
        up = head_mean - hip_mean
    else:
        up = np.array([0, 0, 1.0])

    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        return center, np.eye(3)
    up = up / up_norm

    target = np.array([0, 0, 1.0])
    v = np.cross(up, target)
    s = np.linalg.norm(v)
    c = np.dot(up, target)

    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([-1, 1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    # face camera: L shoulder → +X
    rotated = pts @ R.T
    l_sho_valid = valid[:, 1]
    r_sho_valid = valid[:, 2]
    if l_sho_valid.any() and r_sho_valid.any():
        l_mean = rotated[l_sho_valid, 1].mean(axis=0)
        r_mean = rotated[r_sho_valid, 2].mean(axis=0)
        lr_dir = l_mean - r_mean
        angle = np.arctan2(lr_dir[1], lr_dir[0])
        c2, s2 = np.cos(-angle), np.sin(-angle)
        Rz = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])
        R = Rz @ R

    return center, R


def _apply_transform(joints_seq: np.ndarray, center: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply centering + rotation to joints. Shape: (T, J, 3) → (T, J, 3)."""
    return (joints_seq - center) @ R.T


def draw_skeleton_3d(ax, joints, bones, bone_colors, lw=3.0, alpha=1.0,
                     joint_color="#333", joint_size=40, head_size=80,
                     override_color=None):
    """Draw 3D skeleton. X=right, Y=forward, Z=up."""
    valid = _valid(joints)

    for (i, j) in bones:
        if not valid[i] or not valid[j]:
            continue
        c = override_color or bone_colors.get((i, j), "#666")
        ax.plot(
            [joints[i, 0], joints[j, 0]],
            [joints[i, 1], joints[j, 1]],
            [joints[i, 2], joints[j, 2]],
            color=c, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=2,
        )

    vis_joints = np.where(valid)[0]
    if len(vis_joints) > 0:
        jc = override_color or joint_color
        ax.scatter(
            joints[vis_joints, 0], joints[vis_joints, 1], joints[vis_joints, 2],
            s=joint_size, c=jc, edgecolors="white", linewidths=0.8,
            alpha=alpha, zorder=3, depthshade=False,
        )

    if valid[0]:
        hc = override_color or "#333"
        ax.scatter(
            [joints[0, 0]], [joints[0, 1]], [joints[0, 2]],
            s=head_size, c=hc, edgecolors="white", linewidths=1.2,
            alpha=alpha, zorder=4, depthshade=False,
        )


def visualize_sample(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    pred_action: int,
    gt_action: int,
    sample_idx: int,
    output_dir: Path,
    frames: list[int] | None = None,
):
    T = pred_joints.shape[0]
    if frames is None:
        frames = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    frames = [f for f in frames if f < T]
    n_frames = len(frames)

    # orient both using GT's transform (so they stay aligned)
    center, R = _compute_transform(gt_joints)
    gt_rot = _apply_transform(gt_joints, center, R)
    pred_rot = _apply_transform(pred_joints, center, R)

    pred_name = ACTION_LABELS[pred_action] if pred_action < len(ACTION_LABELS) else "?"
    gt_name = ACTION_LABELS[gt_action] if gt_action < len(ACTION_LABELS) else "?"

    fig = plt.figure(figsize=(6 * n_frames, 8), facecolor="white")
    fig.suptitle(
        f"Sample {sample_idx}  ·  GT: {gt_name}  ·  Pred: {pred_name}",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # compute axis limits from GT
    gt_valid = _valid(gt_rot)
    all_pts = gt_rot[gt_valid]
    if len(all_pts) == 0:
        all_pts = gt_rot.reshape(-1, 3)
    center = all_pts.mean(axis=0)
    span = max(np.abs(all_pts - center).max() * 1.3, 0.2)

    for i, f in enumerate(frames):
        ax = fig.add_subplot(1, n_frames, i + 1, projection="3d")
        ax.set_facecolor("white")

        # GT: color-coded body parts
        draw_skeleton_3d(ax, gt_rot[f], BONES, BONE_COLORS,
                         lw=3.5, alpha=0.6, joint_color="#2d6a4f",
                         joint_size=45, head_size=90)

        # Pred: solid red
        draw_skeleton_3d(ax, pred_rot[f], BONES, BONE_COLORS,
                         lw=3.0, alpha=0.9, override_color="#e74c3c",
                         joint_size=35, head_size=70)

        both_valid = _valid(gt_rot[f]) & _valid(pred_rot[f])
        if both_valid.any():
            err = np.linalg.norm(pred_rot[f] - gt_rot[f], axis=-1)
            frame_mpjpe = err[both_valid].mean() * 1000
        else:
            frame_mpjpe = 0

        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)

        ax.view_init(elev=10, azim=-80)
        ax.set_title(f"t={f}  ({frame_mpjpe:.0f}mm)", fontsize=10, pad=2)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True, alpha=0.1)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("lightgray")
        ax.yaxis.pane.set_edgecolor("lightgray")
        ax.zaxis.pane.set_edgecolor("lightgray")

        if i == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color="#2d6a4f", linewidth=3, label="Ground Truth"),
                Line2D([0], [0], color="#e74c3c", linewidth=3, label="Predicted"),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc="upper left",
                      framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / f"sample_{sample_idx:04d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


@torch.no_grad()
def run_visualization(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    tokenizer, model, cfg = load_model(args.checkpoint, device)
    n_joints = cfg["data"].get("n_joints", 13)
    variant = cfg["model"].get("variant", "base")
    log.info(f"Model: variant={variant}, n_joints={n_joints}")

    gold_dir = args.gold_dir or cfg["data"]["gold_dir"]
    datasets = cfg["data"].get("datasets")

    full = GoldNpzDataset(
        gold_dir, split=None, datasets=datasets,
        augment=False, require_pose=True, require_action=True,
    )

    meta_cache = {}
    def get_split(entry):
        ds = entry["dataset"]
        if ds not in meta_cache:
            mp = Path(gold_dir) / ds / "metadata.npz"
            if mp.exists():
                meta_cache[ds] = np.load(mp, allow_pickle=True)["metadata"]
            else:
                meta_cache[ds] = None
        meta = meta_cache[ds]
        if meta is not None:
            j = entry["index"]
            if j < len(meta):
                return meta[j].get("split", "train")
        return "train"

    test_idx = [i for i, e in enumerate(full.entries) if get_split(e) == "test"]
    if not test_idx:
        n = max(1, len(full) // 5)
        test_idx = list(range(len(full) - n, len(full)))
    log.info(f"Test samples: {len(test_idx)}")

    rng = np.random.RandomState(42)
    sel = rng.choice(test_idx, size=min(args.num_samples, len(test_idx)), replace=False)
    sel.sort()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for count, idx in enumerate(sel):
        batch = full[idx]
        csi = batch["csi"].unsqueeze(0).to(device)
        gt_pose = batch["coords"].numpy()
        gt_action = batch["action_label"].item()

        tokens = tokenizer(csi)
        out = model(tokens)

        if "coords" in out:
            pred_coords = out["coords"][0].cpu().numpy()
        elif "root" in out and "offsets" in out:
            root = out["root"][0].cpu().numpy()
            offsets = out["offsets"][0].cpu().numpy()
            pred_coords = root[:, None, :] + offsets
        else:
            raise KeyError(f"Unknown output keys: {list(out.keys())}")

        action_logits = out["action_logits"][0].cpu()
        pred_action = action_logits.argmax().item()

        if count == 0:
            log.info(f"GT range: [{gt_pose.min():.4f}, {gt_pose.max():.4f}]")
            log.info(f"Pred range: [{pred_coords.min():.4f}, {pred_coords.max():.4f}]")
            valid_count = _valid(gt_pose).sum(axis=1).mean()
            log.info(f"Valid GT joints/frame: {valid_count:.1f}/13")

        per_joint_err = np.linalg.norm(pred_coords - gt_pose, axis=-1)
        mpjpe = per_joint_err.mean() * 1000

        img_path = visualize_sample(
            pred_coords, gt_pose, pred_action, gt_action,
            sample_idx=idx, output_dir=output_dir,
        )

        gt_n = ACTION_LABELS[gt_action] if gt_action < len(ACTION_LABELS) else "?"
        pred_n = ACTION_LABELS[pred_action] if pred_action < len(ACTION_LABELS) else "?"
        log.info(f"[{count+1}/{len(sel)}] idx={idx} MPJPE={mpjpe:.1f}mm GT={gt_n} Pred={pred_n}")

        results.append({
            "idx": int(idx),
            "mpjpe_mm": round(float(mpjpe), 1),
            "gt_action": gt_n,
            "pred_action": pred_n,
            "correct_action": gt_action == pred_action,
        })

    summary_path = output_dir / "results.json"
    avg_mpjpe = float(np.mean([r["mpjpe_mm"] for r in results]))
    action_acc = float(np.mean([r["correct_action"] for r in results]))
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": len(results),
        "avg_mpjpe_mm": round(avg_mpjpe, 1),
        "action_accuracy": round(action_acc, 3),
        "samples": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary: Avg MPJPE={avg_mpjpe:.1f}mm | Action Acc={action_acc:.1%}")
    log.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gold-dir", default=None)
    p.add_argument("--output-dir", default="viz_output")
    p.add_argument("--num-samples", type=int, default=20)
    run_visualization(p.parse_args())
