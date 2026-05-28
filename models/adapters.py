# models/adapters.py

from typing import Dict, Type

import torch
from torch import nn


class BaseAdapter(nn.Module):
    """
    Base CSI adapter.

    The adapter receives flattened per-timestep CSI features:

        [B, T, input_dim]

    and projects them into a shared latent size:

        [B, T, hidden_dim]

    This is only dataset-specific feature alignment.
    It is not a temporal encoder and not a pose head.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape [B, T, input_dim]

        Returns:
            Tensor with shape [B, T, hidden_dim]
        """

        if x.ndim != 3:
            raise ValueError(
                f"BaseAdapter expects [B, T, input_dim], got shape {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {x.shape[-1]}"
            )

        return self.proj(x)


class CSITensorAdapter(BaseAdapter):
    """
    Helper adapter for CSI tensors where the time dimension is not last.

    It:
        1. moves time dimension to index 1
        2. flattens all non-batch, non-time dimensions
        3. applies BaseAdapter projection

    Example:

        Input:
            [B, C, T, A, S]

        After moving time:
            [B, T, C, A, S]

        After flattening:
            [B, T, C * A * S]

        Output:
            [B, T, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        expected_ndim: int,
        time_dim: int = 2,
    ):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)

        self.expected_ndim = expected_ndim
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: dataset-specific CSI tensor

        Returns:
            Tensor with shape [B, T, hidden_dim]
        """

        if x.ndim != self.expected_ndim:
            raise ValueError(
                f"{self.__class__.__name__} expected {self.expected_ndim}D input, "
                f"got shape {tuple(x.shape)}"
            )

        # Move time dimension to position 1:
        # [B, ..., T, ...] -> [B, T, ...]
        dims = list(range(x.ndim))
        time_dim = self.time_dim

        if time_dim < 0:
            time_dim = x.ndim + time_dim

        permute_order = [0, time_dim] + [
            dim for dim in dims if dim not in {0, time_dim}
        ]

        x = x.permute(*permute_order).contiguous()

        batch_size = x.shape[0]
        time_steps = x.shape[1]

        # Flatten every dimension except batch and time:
        # [B, T, ...] -> [B, T, input_dim]
        x = x.view(batch_size, time_steps, -1)

        return super().forward(x)


class WiARAdapter(CSITensorAdapter):
    """
    Adapter for WiAR.

    Input shape:
        [B, 2, T, 3, 3, 30]

    Meaning:
        2  = amplitude + phase channels
        T  = CSI packets / time steps
        3  = TX antennas
        3  = RX antennas
        30 = subcarriers

    Flattened per-timestep feature size:
        2 * 3 * 3 * 30 = 540

    Output shape:
        [B, T, hidden_dim]
    """

    def __init__(self, hidden_dim: int):
        super().__init__(
            input_dim=2 * 3 * 3 * 30,
            hidden_dim=hidden_dim,
            expected_ndim=6,
            time_dim=2,
        )


class WiMANSAdapter(CSITensorAdapter):
    """
    Adapter for WiMANS.

    Input shape:
        [B, 1, T, 3, 3, 30]

    Meaning:
        1  = amplitude channel
        T  = CSI packets / time steps
        3  = TX antennas
        3  = RX antennas
        30 = subcarriers

    Flattened per-timestep feature size:
        1 * 3 * 3 * 30 = 270

    Output shape:
        [B, T, hidden_dim]
    """

    def __init__(self, hidden_dim: int):
        super().__init__(
            input_dim=1 * 3 * 3 * 30,
            hidden_dim=hidden_dim,
            expected_ndim=6,
            time_dim=2,
        )


class WiPoseAdapter(CSITensorAdapter):
    """
    Adapter for Wi-Pose.

    Input shape:
        [B, 1, 5, 3, 3, 30]

    Meaning:
        1  = CSI channel
        5  = CSI packets / time steps
        3  = TX antennas
        3  = RX antennas
        30 = subcarriers

    Flattened per-timestep feature size:
        1 * 3 * 3 * 30 = 270

    Output shape:
        [B, 5, hidden_dim]
    """

    def __init__(self, hidden_dim: int):
        super().__init__(
            input_dim=1 * 3 * 3 * 30,
            hidden_dim=hidden_dim,
            expected_ndim=6,
            time_dim=2,
        )


class PersonWiFiAdapter(CSITensorAdapter):
    """
    Adapter for Person-in-WiFi-3D.

    Input shape:
        [B, 2, 20, 3, 3, 30]

    Meaning:
        2  = amplitude + phase channels
        20 = CSI packets / time steps
        3  = TX antennas
        3  = RX antennas
        30 = subcarriers

    Flattened per-timestep feature size:
        2 * 3 * 3 * 30 = 540

    Output shape:
        [B, 20, hidden_dim]
    """

    def __init__(self, hidden_dim: int):
        super().__init__(
            input_dim=2 * 3 * 3 * 30,
            hidden_dim=hidden_dim,
            expected_ndim=6,
            time_dim=2,
        )


class MMFiAdapter(CSITensorAdapter):
    """
    Adapter for MM-Fi WiFi CSI.

    Input shape:
        [B, 2, T, 3, 114]

    Meaning:
        2   = amplitude + phase channels
        T   = temporal window
        3   = RX antennas
        114 = subcarriers

    Flattened per-timestep feature size:
        2 * 3 * 114 = 684

    Output shape:
        [B, T, hidden_dim]
    """

    def __init__(self, hidden_dim: int):
        super().__init__(
            input_dim=2 * 3 * 114,
            hidden_dim=hidden_dim,
            expected_ndim=5,
            time_dim=2,
        )


class UTHARAdapter(CSITensorAdapter):
    """
    Adapter for UT-HAR.

    Input shape:
        [B, 1, T, 3, 30]

    Meaning:
        1  = amplitude channel
        T  = time steps
        3  = antennas
        30 = subcarriers

    Flattened per-timestep feature size:
        1 * 3 * 30 = 90

    Output shape:
        [B, T, hidden_dim]
    """

    def __init__(self, hidden_dim: int):
        super().__init__(
            input_dim=1 * 3 * 30,
            hidden_dim=hidden_dim,
            expected_ndim=5,
            time_dim=2,
        )


_ADAPTERS: Dict[str, Type[nn.Module]] = {
    "wiar": WiARAdapter,
    "wimans": WiMANSAdapter,
    "wi-pose": WiPoseAdapter,
    "wipose": WiPoseAdapter,
    "person-in-wifi": PersonWiFiAdapter,
    "person_wifi": PersonWiFiAdapter,
    "personwifi": PersonWiFiAdapter,
    "person-in-wifi-3d": PersonWiFiAdapter,
    "mm-fi": MMFiAdapter,
    "mmfi": MMFiAdapter,
    "ut-har": UTHARAdapter,
    "uthar": UTHARAdapter,
}


def get_adapter(dataset_name: str, hidden_dim: int) -> nn.Module:
    """
    Create a dataset-specific adapter.

    Args:
        dataset_name:
            Dataset name such as:
                "wiar"
                "wimans"
                "wipose"
                "person-in-wifi"
                "mmfi"
                "uthar"

        hidden_dim:
            Shared feature dimension.

    Returns:
        Adapter module that outputs [B, T, hidden_dim].
    """

    key = dataset_name.lower().replace("_", "-")

    if key not in _ADAPTERS:
        valid_names = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(
            f"Unknown dataset_name={dataset_name}. Valid names: {valid_names}"
        )

    return _ADAPTERS[key](hidden_dim=hidden_dim)


if __name__ == "__main__":
    batch_size = 4
    hidden_dim = 128

    fake_inputs = {
        "WiAR": (
            WiARAdapter(hidden_dim),
            torch.randn(batch_size, 2, 300, 3, 3, 30),
        ),
        "WiMANS": (
            WiMANSAdapter(hidden_dim),
            torch.randn(batch_size, 1, 3000, 3, 3, 30),
        ),
        "Wi-Pose": (
            WiPoseAdapter(hidden_dim),
            torch.randn(batch_size, 1, 5, 3, 3, 30),
        ),
        "Person-in-WiFi": (
            PersonWiFiAdapter(hidden_dim),
            torch.randn(batch_size, 2, 20, 3, 3, 30),
        ),
        "MM-Fi": (
            MMFiAdapter(hidden_dim),
            torch.randn(batch_size, 2, 10, 3, 114),
        ),
        "UT-HAR": (
            UTHARAdapter(hidden_dim),
            torch.randn(batch_size, 1, 250, 3, 30),
        ),
    }

    for name, (adapter, x) in fake_inputs.items():
        y = adapter(x)
        print(f"{name} -> {list(y.shape)}")

    adapter = get_adapter("mmfi", hidden_dim=hidden_dim)
    x = torch.randn(batch_size, 2, 10, 3, 114)
    y = adapter(x)
    print(f"get_adapter('mmfi') -> {list(y.shape)}")