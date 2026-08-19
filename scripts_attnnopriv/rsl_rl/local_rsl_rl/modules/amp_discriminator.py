from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    """Least-squares AMP discriminator over pairs of AMP observations."""

    def __init__(
        self,
        input_dim: int = 60,
        hidden_dims: Sequence[int] = (1024, 512),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, amp_obs: torch.Tensor, amp_next_obs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([amp_obs, amp_next_obs], dim=-1)
        return self.net(x).squeeze(-1)
