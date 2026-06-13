"""Wi-Mose network — faithful implementation of:
'From Point to Space: 3D Moving Human Pose Estimation Using Commodity WiFi'
(Wang et al., arXiv:2012.14066v1, 2020).

Architecture:
    Input  : (B, 2, H=N_sub, W=T)  — CSI "image" (amplitude + phase channels)
    Body   : input projection (2→4) + 13 bottleneck ResBlocks with progressive
             stride-2 downsampling + AdaptiveAvgPool2d(1,1)
    Head   : FC 2048→512 (ReLU) → FC 512→n_joints×3
    Output : (B, n_joints, 3)  — absolute joint coordinates (metres or mm)

Loss (use together in training):
    L_total = MSE(pred, gt) + Huber(pred, gt, delta=0.75)

Channel / stride schedule (Table I from paper, adapted: in_ch=2 instead of 4):
    Block  in→out   stride
    ─────────────────────
      1    4 →  4     1
      2    4 →  8     1
      3    8 →  8     2  ↓
      4    8 → 16     1
      5   16 → 16     2  ↓
      6   16 → 64     1
      7   64 → 64     2  ↓
      8   64 →256     1
      9  256 →256     2  ↓
     10  256 →1024    1
     11 1024→1024     2  ↓
     12 1024→2048     1
     13 2048→2048     1
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# H36M BFS traversal order for FK (non-root joints)
H36M_BFS_ORDER: list[int] = [1, 4, 7, 2, 5, 8, 3, 6, 9, 11, 14, 10, 12, 15, 13, 16]


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (w,x,y,z) → rotation matrix."""
    a, b, c, d = q.unbind(-1)
    return torch.stack([
        1 - 2 * (c * c + d * d), 2 * (b * c - d * a),     2 * (b * d + c * a),
        2 * (b * c + d * a),     1 - 2 * (b * b + d * d),  2 * (c * d - b * a),
        2 * (b * d - c * a),     2 * (c * d + b * a),      1 - 2 * (b * b + c * c),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


class ForwardKinematics(nn.Module):
    """Differentiable FK: root position + local quaternions + ref bone offsets → 3D joints."""

    def __init__(self, n_joints: int = 17, root_idx: int = 0, parent_map: dict[int, int] | None = None) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.root_idx = root_idx
        self.bfs_order = H36M_BFS_ORDER if n_joints == 17 else list(range(1, n_joints))
        parent = parent_map if parent_map is not None else H36M_PARENT
        parent_list = [-1] * n_joints
        for child, par in parent.items():
            parent_list[child] = par
        self.register_buffer("parent", torch.tensor(parent_list, dtype=torch.long))

    def forward(
        self,
        root_pos: torch.Tensor,      # (B, T, 3)
        quaternions: torch.Tensor,   # (B, T, J-1, 4)
        ref_offsets: torch.Tensor,   # (J, 3)
    ) -> torch.Tensor:
        b, t, _ = root_pos.shape
        device, dtype = root_pos.device, root_pos.dtype
        joints = torch.zeros(b, t, self.n_joints, 3, device=device, dtype=dtype)
        joints[:, :, self.root_idx] = root_pos
        quat_idx = 0
        for joint_i in self.bfs_order:
            par_i = int(self.parent[joint_i].item())
            q = quaternions[:, :, quat_idx]
            r = _quaternion_to_rotation_matrix(q)
            offset = ref_offsets[joint_i].to(dtype=dtype)
            rotated = torch.einsum("btij,j->bti", r, offset)
            joints[:, :, joint_i] = joints[:, :, par_i] + rotated
            quat_idx += 1
        return joints


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Bottleneck residual block: 1×1 → 3×3 → 1×1 with BN+ReLU.

    The 3×3 convolution carries the stride so spatial downsampling happens
    inside the block (same as He et al. v2 pre-activation style but using
    post-activation as in the original Wi-Mose paper description).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # 1×1 compress
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # 3×3 spatial (carries stride)
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # 1×1 expand
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.layers(x) + self.skip(x))


# ---------------------------------------------------------------------------
# Full Wi-Mose network
# ---------------------------------------------------------------------------

_BLOCK_CFG: list[tuple[int, int, int]] = [
    # (in_ch, out_ch, stride)
    (4,    4,    1),   # block  1
    (4,    8,    1),   # block  2
    (8,    8,    2),   # block  3  ↓
    (8,    16,   1),   # block  4
    (16,   16,   2),   # block  5  ↓
    (16,   64,   1),   # block  6
    (64,   64,   2),   # block  7  ↓
    (64,   256,  1),   # block  8
    (256,  256,  2),   # block  9  ↓
    (256,  1024, 1),   # block 10
    (1024, 1024, 2),   # block 11  ↓
    (1024, 2048, 1),   # block 12
    (2048, 2048, 1),   # block 13
]


# ---------------------------------------------------------------------------
# GCN decoder head (topology-aware, replaces flat MLP head)
# ---------------------------------------------------------------------------

class _GCNLayer(nn.Module):
    """Single graph convolution: aggregate neighbours → linear → BN → ReLU."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bn  = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x  : (B, J, in_dim)
        # adj: (J, J) — pre-normalised adjacency on same device
        agg = torch.einsum("bji,jk->bki", x, adj)   # (B, J, in_dim)
        B, J, _ = agg.shape
        out = self.lin(agg)                          # (B, J, out_dim)
        out = self.bn(out.reshape(B * J, -1)).reshape(B, J, -1)
        return F.relu(out, inplace=True)


class WiMoseGCNHead(nn.Module):
    """Topology-aware GCN decoder.

    Instead of predicting all J joints independently via a flat MLP, each joint
    exchanges information with its skeleton neighbours through graph convolution.
    This bakes skeleton connectivity directly into the decoder so that e.g. the
    elbow knows where the shoulder is before predicting its own position.

    Architecture:
        backbone_feat (B, feat_dim)
        → feat_proj  → (B, gcn_dim)   (broadcast to all J joints)
        + joint_embed (J, gcn_dim)     (learnable per-joint token)
        → GCN × n_layers              (topology message-passing)
        → out_head                    (B, J, 3)

    Args:
        feat_dim:   Backbone output channels (2048 for Wi-Mose).
        n_joints:   Number of skeleton joints.
        gcn_dim:    Hidden dimension inside the GCN (default 256).
        n_layers:   Number of GCN message-passing rounds (default 3).
        parent_map: Skeleton parent map; defaults to H36M 17-joint.
    """

    def __init__(
        self,
        feat_dim: int,
        n_joints: int,
        gcn_dim: int = 256,
        n_layers: int = 3,
        parent_map: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.n_joints = n_joints

        pm = parent_map if parent_map is not None else H36M_PARENT

        # ── build normalised adjacency (D^-½ A D^-½) ──────────────────────
        adj = torch.eye(n_joints)
        for child, par in pm.items():
            if child < n_joints and par < n_joints:
                adj[child, par] = 1.0
                adj[par, child] = 1.0   # undirected
        deg = adj.sum(dim=1).clamp(min=1.0)
        d_inv_sqrt = deg.pow(-0.5)
        adj_norm = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
        self.register_buffer("adj", adj_norm)   # (J, J), on correct device auto

        # ── learnable per-joint embeddings ──────────────────────────────────
        self.joint_embed = nn.Embedding(n_joints, gcn_dim)

        # ── project backbone feature → gcn_dim ─────────────────────────────
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, gcn_dim),
            nn.ReLU(inplace=True),
        )

        # ── GCN layers ──────────────────────────────────────────────────────
        self.gcn = nn.ModuleList([_GCNLayer(gcn_dim, gcn_dim) for _ in range(n_layers)])

        # ── per-joint coordinate readout ─────────────────────────────────────
        self.readout = nn.Linear(gcn_dim, 3)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, feat_dim)  — backbone GAP output
        Returns:
            (B, n_joints, 3)
        """
        B = feat.shape[0]

        # broadcast backbone feature + per-joint learnable embedding
        f = self.feat_proj(feat)                       # (B, gcn_dim)
        j = torch.arange(self.n_joints, device=feat.device)
        x = f.unsqueeze(1) + self.joint_embed(j)      # (B, J, gcn_dim)

        for layer in self.gcn:
            x = layer(x, self.adj)                    # (B, J, gcn_dim)

        return self.readout(x)                         # (B, J, 3)


# ---------------------------------------------------------------------------
# Full Wi-Mose network
# ---------------------------------------------------------------------------

class WiMoseNet(nn.Module):
    """Wi-Mose: ResNet backbone → global average pool → head.

    The head can be either:
    - MLP (default, ``use_fk_head=False``, ``use_gcn_head=False``): flat Linear layers.
    - FK (``use_fk_head=True``): predict root + unit quaternions → differentiable FK.
      Bone lengths come from reference skeleton → poses stay human-like by design.
    - GCN (``use_gcn_head=True``): topology-aware graph decoder (legacy).

    Args:
        n_joints:     Number of skeleton joints (17 for H36M, 18 for WiPose).
        in_channels:  CSI channels (default 2: amplitude + phase).
        use_gcn_head: Replace flat MLP head with GCN decoder.
        gcn_dim:      Hidden dim for GCN (used only when use_gcn_head=True).
        gcn_layers:   Number of GCN message-passing rounds.
        parent_map:   Skeleton topology for GCN; defaults to H36M 17-joint.
    """

    def __init__(
        self,
        n_joints: int = 17,
        in_channels: int = 2,
        use_gcn_head: bool = False,
        use_fk_head: bool = False,
        gcn_dim: int = 256,
        gcn_layers: int = 3,
        parent_map: dict[int, int] | None = None,
        num_actions: int = 0,
    ) -> None:
        super().__init__()
        self.n_joints     = n_joints
        self.use_gcn_head = use_gcn_head and not use_fk_head
        self.use_fk_head  = use_fk_head
        self.num_actions  = num_actions

        first_ch = _BLOCK_CFG[0][0]   # 4
        last_ch  = _BLOCK_CFG[-1][1]  # 2048

        # Project dataset channels (2) to network's first channel count (4)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, first_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(first_ch),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[_ResBlock(ic, oc, s) for ic, oc, s in _BLOCK_CFG]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        if use_fk_head:
            hidden = 512
            self.fk_head = nn.Sequential(
                nn.Linear(last_ch, hidden),
                nn.ReLU(inplace=True),
            )
            self.root_head = nn.Linear(hidden, 3)
            self.quat_head = nn.Linear(hidden, (n_joints - 1) * 4)
            self.fk = ForwardKinematics(n_joints)
            self.register_buffer("ref_offsets", torch.zeros(n_joints, 3))
            nn.init.zeros_(self.quat_head.bias)
            with torch.no_grad():
                bias = self.quat_head.bias.reshape(n_joints - 1, 4)
                bias[:, 0] = 1.0  # identity quaternion (w=1)
            self.head = None
        elif self.use_gcn_head:
            self.head = WiMoseGCNHead(
                feat_dim=last_ch,
                n_joints=n_joints,
                gcn_dim=gcn_dim,
                n_layers=gcn_layers,
                parent_map=parent_map,
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(last_ch, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, n_joints * 3),
            )

        self.action_head = nn.Linear(last_ch, num_actions) if num_actions > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_action: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """
        Args:
            x:               (B, C=2, H=N_sub, W=T)
            return_features: If True, also return backbone GAP features (B, 2048)
                             for computing uniformity regularization in the training loop.
        Returns:
            (B, n_joints, 3)  or  ((B, n_joints, 3), (B, 2048)) if return_features=True
        """
        z = self.input_proj(x)      # (B, 4, H, W)
        z = self.blocks(z)           # (B, 2048, H', W')
        feat = self.gap(z).flatten(1)  # (B, 2048)
        if self.use_fk_head:
            h = self.fk_head(feat)
            root = self.root_head(h)                                          # (B, 3)
            quat = F.normalize(
                self.quat_head(h).view(-1, self.n_joints - 1, 4), dim=-1,
            )                                                                 # (B, J-1, 4)
            joints = self.fk(
                root.unsqueeze(1), quat.unsqueeze(1), self.ref_offsets,
            )                                                                 # (B, 1, J, 3)
            coords = joints.squeeze(1)
        elif self.use_gcn_head:
            coords = self.head(feat)                        # (B, J, 3)
        else:
            coords = self.head(feat).view(-1, self.n_joints, 3)  # (B, J, 3)

        if self.action_head is not None and (return_action or self.training):
            out: dict[str, torch.Tensor] = {
                "coords": coords,
                "action_logits": self.action_head(feat),
            }
            if return_features:
                out["features"] = feat
            return out

        if return_features:
            return coords, feat
        return coords


# ---------------------------------------------------------------------------
# Skeleton topology (H36M 17-joint)
# ---------------------------------------------------------------------------

# joint_id → parent_id
H36M_PARENT: dict[int, int] = {
    1: 0, 2: 1, 3: 2,         # right leg:  pelvis→R.hip→R.knee→R.foot
    4: 0, 5: 4, 6: 5,         # left leg:   pelvis→L.hip→L.knee→L.foot
    7: 0, 8: 7,                # torso:      pelvis→spine→neck
    9: 8, 10: 9,               # head:       neck→head→head_top
    11: 8, 12: 11, 13: 12,    # left arm:   neck→L.shoulder→L.elbow→L.wrist
    14: 8, 15: 14, 16: 15,    # right arm:  neck→R.shoulder→R.elbow→R.wrist
}

# (left_child, left_parent, right_child, right_parent) — anatomically symmetric bones
H36M_SYM_PAIRS: list[tuple[int, int, int, int]] = [
    (5, 4,  2, 1),    # thigh   L vs R
    (6, 5,  3, 2),    # shin    L vs R
    (12, 11, 15, 14), # upper arm L vs R
    (13, 12, 16, 15), # forearm   L vs R
    (4,  0,  1, 0),   # hip offset L vs R
    (11, 8,  14, 8),  # shoulder width L vs R
]

# Per-joint loss weights: extremities get 3× weight, hips/shoulders 1.5×.
# Wrists and feet have highest positional error → pushed harder.
H36M_JOINT_WEIGHTS: list[float] = [
    1.0,  # 0  Pelvis (root — always 0 after centering, no gradient info)
    1.5,  # 1  R.Hip
    2.0,  # 2  R.Knee
    3.0,  # 3  R.Foot
    1.5,  # 4  L.Hip
    2.0,  # 5  L.Knee
    3.0,  # 6  L.Foot
    1.0,  # 7  Spine
    1.0,  # 8  Neck
    1.0,  # 9  Head
    1.0,  # 10 Head_top
    1.5,  # 11 L.Shoulder
    2.0,  # 12 L.Elbow
    3.0,  # 13 L.Wrist
    1.5,  # 14 R.Shoulder
    2.0,  # 15 R.Elbow
    3.0,  # 16 R.Wrist
]


def uniformity_loss(features: torch.Tensor) -> torch.Tensor:
    """Uniformity regularisation on backbone feature vectors.

    Encourages the batch of features to spread uniformly on the unit hyper-sphere
    rather than collapsing to a single point (mean-collapse on feature level).

    Wang et al. "Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere", ICML 2020.

    Args:
        features: (B, D) raw backbone GAP features (not L2-normalised yet).
    Returns:
        Scalar loss: log mean(exp(-2 * pairwise_cosine_sq_dist)) — more negative is better,
        we *minimise* the negative → maximise uniformity.
    """
    if features.shape[0] < 2:
        return features.new_tensor(0.0)
    z = torch.nn.functional.normalize(features, dim=-1)  # (B, D)
    sq_dists = torch.cdist(z, z, p=2).pow(2)             # (B, B)
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return sq_dists[mask].mul(-2.0).exp().mean().log()


# ---------------------------------------------------------------------------
# Loss (used in train_wimose.py)
# ---------------------------------------------------------------------------

class WiMoseLoss(nn.Module):
    """Structured pose loss = weighted (MSE+Huber) + bone length + symmetry + diversity.

    Args:
        delta:           Huber δ (default 0.75 per Wi-Mose paper).
        lambda_bone:     Weight for bone-length consistency loss.  0 = disabled.
        lambda_sym:      Weight for left-right symmetry loss.      0 = disabled.
        lambda_div:      Weight for batch diversity loss.           0 = disabled.
        joint_weights:   Per-joint loss weights; defaults to H36M extremity weighting.
        parent_map:      Skeleton parent map; defaults to H36M 17-joint.
        sym_pairs:       Symmetric bone pairs;  defaults to H36M pairs.
    """

    def __init__(
        self,
        delta: float = 0.75,
        lambda_bone: float = 0.5,
        lambda_sym: float = 0.1,
        lambda_div: float = 0.05,
        lambda_collapse: float = 0.0,
        lambda_spread: float = 0.0,
        joint_weights: list[float] | None = None,
        parent_map: dict[int, int] | None = None,
        sym_pairs: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.huber       = nn.HuberLoss(reduction="none", delta=delta)
        self.huber_delta = delta
        self.lambda_bone     = lambda_bone
        self.lambda_sym      = lambda_sym
        self.lambda_div      = lambda_div
        self.lambda_collapse = lambda_collapse
        self.lambda_spread   = lambda_spread
        self.parent_map  = parent_map if parent_map is not None else H36M_PARENT
        self.sym_pairs   = sym_pairs  if sym_pairs  is not None else H36M_SYM_PAIRS

        jw = joint_weights if joint_weights is not None else H36M_JOINT_WEIGHTS
        # register as buffer so it moves with .to(device) automatically
        self.register_buffer("joint_w", torch.tensor(jw, dtype=torch.float32))

    # ── structural terms ────────────────────────────────────────────────────

    def _bone_length_loss(
        self,
        pred: torch.Tensor,   # (B, J, 3)
        gt: torch.Tensor,     # (B, J, 3)
    ) -> torch.Tensor:
        """L1 error between predicted and GT bone lengths, averaged over bones."""
        loss = pred.new_tensor(0.0)
        for child, par in self.parent_map.items():
            pred_bl = (pred[:, child] - pred[:, par]).norm(dim=-1)   # (B,)
            gt_bl   = (gt[:, child]   - gt[:, par]).norm(dim=-1)     # (B,)
            loss = loss + (pred_bl - gt_bl).abs().mean()
        return loss / len(self.parent_map)

    def _symmetry_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """L1 difference between symmetric left/right bone lengths (no GT needed)."""
        loss = pred.new_tensor(0.0)
        for lc, lp, rc, rp in self.sym_pairs:
            left_bl  = (pred[:, lc] - pred[:, lp]).norm(dim=-1)
            right_bl = (pred[:, rc] - pred[:, rp]).norm(dim=-1)
            loss = loss + (left_bl - right_bl).abs().mean()
        return loss / len(self.sym_pairs)

    def _diversity_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise batch variance collapse: push pred variance toward GT variance."""
        if pred.shape[0] < 2:
            return pred.new_tensor(0.0)
        pred_var = pred.var(dim=0)           # (J, 3)
        gt_var   = gt.var(dim=0).detach()    # (J, 3) — no grad through GT
        return (pred_var - gt_var).abs().mean()

    def _collapse_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Hinge on std shortfall: penalise pred batch std << GT batch std per joint."""
        if pred.shape[0] < 2:
            return pred.new_tensor(0.0)
        pred_std = pred.std(dim=0)
        gt_std   = gt.std(dim=0).detach()
        shortfall = F.relu(gt_std - pred_std)
        return (shortfall / (gt_std + 1e-6)).mean()

    def _spread_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """When GT poses differ in batch, predicted poses must also spread apart."""
        if pred.shape[0] < 2:
            return pred.new_tensor(0.0)
        b = pred.shape[0]
        pred_flat = pred.reshape(b, -1)
        gt_flat   = gt.reshape(b, -1)
        gt_dist   = torch.cdist(gt_flat, gt_flat)
        pred_dist = torch.cdist(pred_flat, pred_flat)
        mask = ~torch.eye(b, dtype=torch.bool, device=pred.device)
        gap = F.relu(gt_dist[mask] - pred_dist[mask])
        return gap.mean() / (gt_dist[mask].mean() + 1e-6)

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        pred: torch.Tensor,                # (B, J, 3)
        target: torch.Tensor,              # (B, J, 3)
        mask: torch.Tensor | None = None,  # (B,) float — 1 if pose valid
    ) -> torch.Tensor:
        # per-joint weighting: w (J,) → (1, J, 1) broadcast
        jw = self.joint_w
        if jw.shape[0] != pred.shape[1]:
            # fall back to uniform weights if joint count mismatch (e.g. WiPose 18j)
            jw = jw.new_ones(pred.shape[1])
        jw = jw.view(1, -1, 1)  # (1, J, 1)

        diff = pred - target  # (B, J, 3)

        # Weighted L2
        l2_per = diff.pow(2) * jw   # (B, J, 3)
        # Weighted Huber
        huber_per = self.huber(pred, target) * jw  # (B, J, 3)

        if mask is not None:
            m = mask.view(-1, 1, 1).float()
            n_valid = m.sum().clamp(min=1)
            denom = n_valid * pred.shape[1] * pred.shape[2]
            base_loss = (l2_per * m).sum() / denom + (huber_per * m).sum() / denom
        else:
            base_loss = l2_per.mean() + huber_per.mean()

        bone_loss     = self._bone_length_loss(pred, target) * self.lambda_bone
        sym_loss      = self._symmetry_loss(pred)             * self.lambda_sym
        div_loss      = self._diversity_loss(pred, target)    * self.lambda_div
        collapse_loss = self._collapse_loss(pred, target)     * self.lambda_collapse
        spread_loss   = self._spread_loss(pred, target)       * self.lambda_spread

        return base_loss + bone_loss + sym_loss + div_loss + collapse_loss + spread_loss

    @staticmethod
    def n_coords(t: torch.Tensor) -> int:
        return t.shape[1] * t.shape[2]  # J * 3


def compute_ref_offsets(
    dataset,
    n_joints: int = 17,
    root_joint: int = 0,
    n_sample: int = 512,
    seed: int = 42,
) -> torch.Tensor:
    """Mean bone offsets (child − parent) from training poses for FK decoder."""
    parent_map = H36M_PARENT if n_joints == 17 else {
        i: max(0, i - 1) for i in range(1, n_joints)
    }

    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[: min(n_sample, n)].tolist()

    acc = torch.zeros(n_joints, 3, dtype=torch.float64)
    for i in idx:
        coords = dataset[i]["coords"]
        gt = coords[coords.shape[0] // 2].double()
        if 0 <= root_joint < gt.shape[0]:
            gt = gt - gt[root_joint : root_joint + 1]
        acc += gt
    ref = acc / max(len(idx), 1)

    offsets = torch.zeros(n_joints, 3, dtype=torch.float32)
    for child, par in parent_map.items():
        offsets[child] = (ref[child] - ref[par]).float()
    return offsets


@torch.no_grad()
def diversity_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """Batch-level collapse diagnostics (higher std_ratio / spread_ratio = better)."""
    if pred.shape[0] < 2:
        return {"std_ratio": 1.0, "spread_ratio": 1.0}
    pred_std = pred.std(dim=0).mean().item()
    gt_std   = gt.std(dim=0).mean().item()
    b = pred.shape[0]
    pf = pred.reshape(b, -1)
    gf = gt.reshape(b, -1)
    mask = ~torch.eye(b, dtype=torch.bool, device=pred.device)
    spread_ratio = (
        torch.cdist(pf, pf)[mask].mean() / (torch.cdist(gf, gf)[mask].mean() + 1e-8)
    ).item()
    return {
        "std_ratio": pred_std / (gt_std + 1e-8),
        "spread_ratio": spread_ratio,
    }
