"""WiPose: 3D Human Pose Construction Using WiFi (MobiCom 2020).

Faithful reimplementation of Jiang et al.
Architecture: 4-layer 2D CNN → 3-layer LSTM → Forward Kinematics (quaternion).

Reference: https://doi.org/10.1145/3372224.3380900
"""

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Wi-Pose 18-joint skeleton tree (VICON markers)
# Root = Neck (index 1)
# ---------------------------------------------------------------------------
WIPOSE_ROOT = 1
WIPOSE_PARENT = {
    0: 1,                       # Head ← Neck
    2: 1, 3: 2, 4: 3, 5: 4,    # R arm chain
    6: 1, 7: 6, 8: 7, 9: 8,    # L arm chain
    10: 1, 11: 10, 12: 11, 13: 12,  # R leg chain
    14: 1, 15: 14, 16: 15, 17: 16,  # L leg chain
}
WIPOSE_BFS_ORDER = [0, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16, 5, 9, 13, 17]

# ---------------------------------------------------------------------------
# Human3.6M 17-joint skeleton tree (MM-Fi ground truth ordering)
# Root = Pelvis (index 0)
# ---------------------------------------------------------------------------
H36M_ROOT = 0
H36M_PARENT = {
    1: 0, 2: 1, 3: 2,              # R leg: Pelvis→R_Hip→R_Knee→R_Ankle
    4: 0, 5: 4, 6: 5,              # L leg: Pelvis→L_Hip→L_Knee→L_Ankle
    7: 0, 8: 7,                    # Spine: Pelvis→Spine→Thorax
    9: 8, 10: 9,                   # Head: Thorax→Nose→Head_Top
    11: 8, 12: 11, 13: 12,         # L arm: Thorax→L_Shoulder→L_Elbow→L_Wrist
    14: 8, 15: 14, 16: 15,         # R arm: Thorax→R_Shoulder→R_Elbow→R_Wrist
}
H36M_BFS_ORDER = [1, 4, 7, 2, 5, 8, 3, 6, 9, 11, 14, 10, 12, 15, 13, 16]

SKELETON_CONFIGS = {
    18: (WIPOSE_ROOT, WIPOSE_PARENT, WIPOSE_BFS_ORDER),
    17: (H36M_ROOT, H36M_PARENT, H36M_BFS_ORDER),
}


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions to rotation matrices.

    Args:
        q: (..., 4) unit quaternions (a, b, c, d) = (w, x, y, z)
    Returns:
        R: (..., 3, 3) rotation matrices
    """
    a, b, c, d = q.unbind(-1)

    R = torch.stack([
        1 - 2*(c*c + d*d),  2*(b*c - d*a),      2*(b*d + c*a),
        2*(b*c + d*a),      1 - 2*(b*b + d*d),   2*(c*d - b*a),
        2*(b*d - c*a),      2*(c*d + b*a),        1 - 2*(b*b + c*c),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    return R


class ForwardKinematics(nn.Module):
    """Differentiable FK layer.

    Given a root position and per-joint quaternion rotations,
    recursively computes joint positions using reference bone offsets.
    """

    def __init__(self, n_joints: int = 18):
        super().__init__()
        self.n_joints = n_joints

        root_idx, parent_map, bfs_order = SKELETON_CONFIGS[n_joints]
        self.root_idx = root_idx
        self.bfs_order = bfs_order

        parent_list = [-1] * n_joints
        for child, parent in parent_map.items():
            parent_list[child] = parent
        self.register_buffer("parent", torch.tensor(parent_list, dtype=torch.long))

    def forward(
        self,
        root_pos: torch.Tensor,
        quaternions: torch.Tensor,
        ref_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            root_pos:    (B, T, 3)  — predicted root joint position
            quaternions: (B, T, N-1, 4) — unit quaternions per non-root joint
            ref_offsets: (N, 3) — reference bone offsets (child - parent in rest pose)
        Returns:
            joints: (B, T, N, 3) — reconstructed joint positions
        """
        B, T, _ = root_pos.shape
        N = self.n_joints
        device = root_pos.device

        joints = torch.zeros(B, T, N, 3, device=device, dtype=root_pos.dtype)
        joints[:, :, self.root_idx] = root_pos

        quat_idx = 0
        for joint_i in self.bfs_order:
            parent_i = self.parent[joint_i].item()

            q = quaternions[:, :, quat_idx]  # (B, T, 4)
            R = _quaternion_to_rotation_matrix(q)  # (B, T, 3, 3)

            offset = ref_offsets[joint_i]  # (3,)
            rotated = torch.einsum("btij,j->bti", R, offset)  # (B, T, 3)

            joints[:, :, joint_i] = joints[:, :, parent_i] + rotated
            quat_idx += 1

        return joints


class WiPoseEncoder(nn.Module):
    """4-layer 2D CNN encoder (paper Section 3.2).

    Input per frame: (9, 30, P) where 9=antennas, 30=subcarriers, P=packets.
    Filters: [64, 128, 64, 1] with BatchNorm + LeakyReLU + Dropout.
    """

    def __init__(
        self,
        in_channels: int = 9,
        n_sub: int = 30,
        n_packets: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_sub = n_sub
        self.n_packets = n_packets

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Dropout2d(dropout),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.02),
            nn.Dropout2d(dropout),
        )
        self.out_features = n_sub * n_packets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, 9, 30, P) — absolute CSI amplitude
        Returns:
            z: (B*T, out_features) — flattened feature vector
        """
        h = self.layers(x)  # (B*T, 1, 30, P)
        return h.reshape(h.size(0), -1)  # (B*T, 30*P)


class WiPoseNet(nn.Module):
    """Full WiPose model: CNN → LSTM → FK.

    Paper hyperparameters:
        CNN filters: [64, 128, 64, 1], CNN dropout: 0.2, LeakyReLU alpha: 0.02
        LSTM: 3 layers, hidden_size=544, dropout=0.1
        Loss: Lp + Ls + Lr with β=γ=1.0
    """

    def __init__(
        self,
        n_joints: int = 18,
        n_antennas: int = 9,
        n_sub: int = 30,
        n_packets: int = 5,
        lstm_hidden: int = 544,
        lstm_layers: int = 3,
        lstm_dropout: float = 0.1,
        cnn_dropout: float = 0.2,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.n_antennas = n_antennas
        self.n_sub = n_sub
        self.n_packets = n_packets

        self.encoder = WiPoseEncoder(
            in_channels=n_antennas,
            n_sub=n_sub,
            n_packets=n_packets,
            dropout=cnn_dropout,
        )

        self.lstm = nn.LSTM(
            input_size=self.encoder.out_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.root_head = nn.Linear(lstm_hidden, 3)
        self.quat_head = nn.Linear(lstm_hidden, (n_joints - 1) * 4)

        self.fk = ForwardKinematics(n_joints)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.02, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.quat_head.bias)
        with torch.no_grad():
            bias = self.quat_head.bias.reshape(-1, 4)
            bias[:, 0] = 1.0  # identity quaternion (1, 0, 0, 0)

    def forward(
        self,
        csi: torch.Tensor,
        ref_offsets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            csi:         (B, T, 9, 30, P) — absolute CSI amplitude per frame
            ref_offsets: (18, 3) — reference bone offsets for FK
        Returns:
            dict with:
                coords:      (B, T, 18, 3) — predicted joint positions
                root_pos:    (B, T, 3)
                quaternions: (B, T, 17, 4)
        """
        B, T = csi.shape[:2]

        x = csi.reshape(B * T, self.n_antennas, self.n_sub, self.n_packets)
        z = self.encoder(x)  # (B*T, feat_dim)
        z = z.reshape(B, T, -1)  # (B, T, feat_dim)

        lstm_out, _ = self.lstm(z)  # (B, T, hidden)

        root_pos = self.root_head(lstm_out)  # (B, T, 3)

        quat_raw = self.quat_head(lstm_out)  # (B, T, 17*4)
        quat_raw = quat_raw.reshape(B, T, self.n_joints - 1, 4)
        quaternions = F.normalize(quat_raw, dim=-1)  # unit quaternions

        joints = self.fk(root_pos, quaternions, ref_offsets)  # (B, T, 18, 3)

        return {
            "coords": joints,
            "root_pos": root_pos,
            "quaternions": quaternions,
        }
