"""Structure-aware CTS-MoE policy modules.

This file intentionally contains only PyTorch modules.  The classes here do not
depend on IsaacLab managers or the current PPO runner, so they can be plugged
into the existing training code in a later step without changing rollout logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _make_activation(activation: str | type[nn.Module] | nn.Module) -> nn.Module:
    """Create an activation module from a small set of common names."""
    if isinstance(activation, nn.Module):
        return activation
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()

    name = activation.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name in ("identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {activation}")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str | type[nn.Module] | nn.Module = "elu",
    output_activation: str | type[nn.Module] | nn.Module = "identity",
) -> nn.Sequential:
    """Build a simple fully-connected network."""
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(_make_activation(activation))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    layers.append(_make_activation(output_activation))
    return nn.Sequential(*layers)


def build_lazy_mlp(
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str | type[nn.Module] | nn.Module = "elu",
    output_activation: str | type[nn.Module] | nn.Module = "identity",
) -> nn.Sequential:
    """Build an MLP whose first Linear infers its input dimension at runtime."""
    layers: list[nn.Module] = []
    if hidden_dims:
        layers.append(nn.LazyLinear(hidden_dims[0]))
        layers.append(_make_activation(activation))
        for idx in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            layers.append(_make_activation(activation))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
    else:
        layers.append(nn.LazyLinear(output_dim))
    layers.append(_make_activation(output_activation))
    return nn.Sequential(*layers)


def batched_gram_schmidt(U: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable batched Gram-Schmidt orthogonalization.

    Args:
        U: Expert feature tensor with shape [B, K, D].
        eps: Numerical stabilizer for projection and normalization.

    Returns:
        Orthogonalized and L2-normalized features with shape [B, K, D].
    """
    if U.dim() != 3:
        raise ValueError(f"U must be [B, K, D], got shape {tuple(U.shape)}")

    vectors = []
    for i in range(U.shape[1]):
        v = U[:, i, :]
        for prev in vectors:
            proj_coeff = (v * prev).sum(dim=-1, keepdim=True) / ((prev * prev).sum(dim=-1, keepdim=True) + eps)
            v = v - proj_coeff * prev
        v = v / (v.norm(dim=-1, keepdim=True) + eps)
        vectors.append(v)
    return torch.stack(vectors, dim=1)


def compute_orthogonality_metrics(
    V: torch.Tensor,
    gate_weights: torch.Tensor | None = None,
    gate_activation: str = "softmax",
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute actor-side orthogonality and optional gate usage metrics."""
    if V.dim() != 3:
        raise ValueError(f"V must be [B, K, D], got shape {tuple(V.shape)}")

    num_experts = V.shape[1]
    V_norm = F.normalize(V, dim=-1, eps=eps)
    gram = torch.matmul(V_norm, V_norm.transpose(-1, -2))
    eye = torch.eye(num_experts, device=V.device, dtype=V.dtype).unsqueeze(0)
    off_diag_mask = ~torch.eye(num_experts, device=V.device, dtype=torch.bool)
    off_diag_abs = (gram - eye).abs()[:, off_diag_mask]
    diag = gram.diagonal(dim1=-2, dim2=-1)

    metrics = {
        "actor/orth_offdiag_mean_abs": off_diag_abs.mean(),
        "actor/orth_offdiag_max_abs": off_diag_abs.max(),
        "actor/orth_diag_mean": diag.mean(),
        "actor/orth_diag_std": diag.std(unbiased=False),
    }

    if gate_weights is not None:
        if gate_weights.dim() != 2 or gate_weights.shape[1] != num_experts:
            raise ValueError(f"gate_weights must be [B, {num_experts}], got shape {tuple(gate_weights.shape)}")
        gate_coeffs = gate_weights
        gate_mean = gate_coeffs.mean(dim=0)
        gate_std = gate_coeffs.std(dim=0, unbiased=False)
        gate_abs = gate_coeffs.abs()
        gate_abs_mean = gate_abs.mean(dim=0)
        gate_abs_max = gate_abs.max(dim=0).values
        metrics["actor/gate_coeff_global_abs_mean"] = gate_abs.mean()
        metrics["actor/gate_coeff_global_abs_max"] = gate_abs.max()
        for expert_idx in range(num_experts):
            metrics[f"actor/gate_coeff_mean_{expert_idx}"] = gate_mean[expert_idx]
            metrics[f"actor/gate_coeff_std_{expert_idx}"] = gate_std[expert_idx]
            metrics[f"actor/gate_coeff_abs_mean_{expert_idx}"] = gate_abs_mean[expert_idx]
            metrics[f"actor/gate_coeff_abs_max_{expert_idx}"] = gate_abs_max[expert_idx]

        if gate_activation == "softmax":
            gate_entropy = -(gate_coeffs * torch.log(gate_coeffs + eps)).sum(dim=-1)
            metrics["actor/gate_entropy"] = gate_entropy.mean()
            for expert_idx in range(num_experts):
                metrics[f"actor/gate_usage_mean_{expert_idx}"] = gate_mean[expert_idx]
                metrics[f"actor/gate_usage_std_{expert_idx}"] = gate_std[expert_idx]
        elif gate_activation == "tanh":
            positive_ratio = (gate_coeffs > 0.0).float().mean(dim=0)
            negative_ratio = (gate_coeffs < 0.0).float().mean(dim=0)
            saturation_ratio = (gate_abs > 0.95).float().mean(dim=0)
            for expert_idx in range(num_experts):
                metrics[f"actor/gate_positive_ratio_{expert_idx}"] = positive_ratio[expert_idx]
                metrics[f"actor/gate_negative_ratio_{expert_idx}"] = negative_ratio[expert_idx]
                metrics[f"actor/gate_saturation_ratio_{expert_idx}"] = saturation_ratio[expert_idx]

    return metrics


class GridEncoder(nn.Module):
    """Small CNN for height maps, occupancy grids, depth maps, or LiDAR grids."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        filters: Sequence[int] = (16, 32, 64),
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_channels = in_channels
        for out_channels in filters:
            layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(_make_activation(activation))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = out_channels

        # Adaptive pooling keeps the module agnostic to the exact ray-grid size.
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_channels, output_dim))
        layers.append(_make_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"GridEncoder expects [B, C, H, W], got shape {tuple(x.shape)}")
        return self.net(x)


class HeightScanCNNEncoder(nn.Module):
    """CNN for stacked height-scan rays shaped as [B, C, num_rays, 1]."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        filters: Sequence[int] = (16, 32, 64),
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_channels = in_channels
        for out_channels in filters:
            layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)))
            layers.append(_make_activation(activation))
            layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True))
            prev_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_channels, output_dim))
        layers.append(_make_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"HeightScanCNNEncoder expects [B, C, H, W], got shape {tuple(x.shape)}")
        return self.net(x)


class TeacherEncoder(nn.Module):
    """Teacher encoder using privileged vector, height map, and optional context.

    Height scan channel convention for the Go2Piper scene:
      height_scan[:, 0] = H_ground_scan
      height_scan[:, 1] = H_lateral_scan
      height_scan[:, 2] = H_overhead_scan
    """

    def __init__(
        self,
        privileged_dim: int,
        latent_dim: int = 32,
        height_channels: int = 3,
        num_tasks: int = 4,
        context_dim: int = 0,
        height_flat_dim: int | None = None,
        semantic_decoupled: bool = False,
        height_feature_dim: int = 128,
        privileged_feature_dim: int = 32,
        height_encoder_type: str = "mlp",
        height_hidden_dims: Sequence[int] = (512, 256),
        height_cnn_filters: Sequence[int] = (16, 32, 64),
        privileged_hidden_dims: Sequence[int] = (512, 256),
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.height_channels = height_channels
        self.num_tasks = num_tasks
        self.context_dim = context_dim
        self.semantic_decoupled = semantic_decoupled
        self.height_encoder_type = height_encoder_type.lower()

        if self.height_encoder_type == "mlp":
            # h_t is the flattened multi-layer height map. LazyLinear keeps this
            # independent of the exact ray grid resolution until config wiring is done.
            if height_flat_dim is None:
                self.height_encoder = build_lazy_mlp(
                    hidden_dims=height_hidden_dims,
                    output_dim=height_feature_dim,
                    activation=activation,
                )
            else:
                self.height_encoder = build_mlp(
                    height_flat_dim,
                    hidden_dims=height_hidden_dims,
                    output_dim=height_feature_dim,
                    activation=activation,
                )
        elif self.height_encoder_type == "cnn":
            self.height_encoder = HeightScanCNNEncoder(
                in_channels=height_channels,
                output_dim=height_feature_dim,
                filters=height_cnn_filters,
                activation=activation,
            )
        else:
            raise ValueError("height_encoder_type must be 'mlp' or 'cnn'")

        self.privileged_encoder = build_mlp(
            privileged_dim,
            hidden_dims=privileged_hidden_dims,
            output_dim=privileged_feature_dim,
            activation=activation,
        )
        self.projection = nn.Sequential(
            nn.Linear(context_dim + privileged_feature_dim + height_feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.task_classifier = nn.Linear(latent_dim, num_tasks)

    def forward(
        self,
        height_scan: torch.Tensor,
        privileged_obs: torch.Tensor,
        context: torch.Tensor | None = None,
        return_task_id: bool = False,
        return_task_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if height_scan.dim() != 4:
            raise ValueError(f"height_scan must be [B, C_h, H, W], got {tuple(height_scan.shape)}")
        if height_scan.shape[1] != self.height_channels:
            raise ValueError(
                f"Expected {self.height_channels} height channels, got {height_scan.shape[1]}"
            )
        if privileged_obs.dim() != 2:
            raise ValueError(f"privileged_obs must be [B, priv_dim], got {tuple(privileged_obs.shape)}")

        if self.height_encoder_type == "mlp":
            height_feature = self.height_encoder(height_scan.flatten(start_dim=1))
        else:
            height_feature = self.height_encoder(height_scan)
        privileged_feature = self.privileged_encoder(privileged_obs)
        projection_inputs = []
        if self.context_dim > 0:
            if context is None:
                raise ValueError("context is required when context_dim > 0")
            if context.dim() != 2 or context.shape[-1] != self.context_dim:
                raise ValueError(f"context must be [B, {self.context_dim}], got {tuple(context.shape)}")
            projection_inputs.append(context)
        projection_inputs.extend([privileged_feature, height_feature])
        z_teacher = self.projection(torch.cat(projection_inputs, dim=-1))

        if not return_task_id and not return_task_logits:
            return z_teacher

        task_logits = self.task_classifier(z_teacher)
        task_id = torch.argmax(task_logits, dim=-1)
        if return_task_logits:
            return z_teacher, task_id, task_logits
        return z_teacher, task_id


class StudentEncoder(nn.Module):
    """Student encoder using proprioception history and raycaster depth image."""

    def __init__(
        self,
        proprio_dim: int,
        latent_dim: int = 32,
        perception_type: str = "depth",
        perception_dim: int | None = None,
        perception_channels: int = 1,
        proprio_feature_dim: int = 32,
        proprio_hidden_dims: Sequence[int] = (512, 256),
        depth_feature_dim: int = 128,
        depth_filters: Sequence[int] = (16, 32, 64),
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 1,
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        if perception_type not in ("depth", "grid", "vector"):
            raise ValueError("perception_type must be 'depth', 'grid', or 'vector'")
        if perception_type == "vector" and perception_dim is None:
            raise ValueError("perception_dim is required when perception_type='vector'")

        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.perception_type = perception_type

        # MLP(o^p_{t-H:t}) -> 32.  The history is flattened before this MLP.
        self.proprio_history_encoder = build_lazy_mlp(
            hidden_dims=proprio_hidden_dims,
            output_dim=proprio_feature_dim,
            activation=activation,
        )

        if perception_type in ("depth", "grid"):
            self.perception_encoder = GridEncoder(
                in_channels=perception_channels,
                output_dim=depth_feature_dim,
                filters=depth_filters,
                activation=activation,
            )
        else:
            self.perception_encoder = build_mlp(
                perception_dim,
                hidden_dims=proprio_hidden_dims,
                output_dim=depth_feature_dim,
                activation=activation,
            )

        self.temporal_gru = nn.GRU(
            input_size=depth_feature_dim + proprio_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden_dim + proprio_feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, proprio_history: torch.Tensor, perception: torch.Tensor) -> torch.Tensor:
        if proprio_history.dim() != 3:
            raise ValueError(
                f"proprio_history must be [B, T, proprio_dim], got {tuple(proprio_history.shape)}"
            )
        if proprio_history.shape[-1] != self.proprio_dim:
            raise ValueError(f"Expected proprio_dim={self.proprio_dim}, got {proprio_history.shape[-1]}")

        proprio_history_feature = self.proprio_history_encoder(proprio_history.flatten(start_dim=1))
        depth_feature = self.perception_encoder(perception)
        current_proprio = proprio_history[:, -1]
        temporal_input = torch.cat([depth_feature, current_proprio], dim=-1).unsqueeze(1)
        _, h_n = self.temporal_gru(temporal_input)
        temporal_feature = h_n[-1]
        return self.projection(torch.cat([temporal_feature, proprio_history_feature], dim=-1))


class MoEActor(nn.Module):
    """Dense Mixture-of-Experts actor with soft routing."""

    DEFAULT_EXPERT_NAMES = (
        "lateral_avoidance",
        "under_table",
        "stair_up",
        "flat",
    )

    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_experts: int = 4,
        expert_hidden_dims: Sequence[int] = (256, 128),
        router_hidden_dims: Sequence[int] = (128, 64),
        expert_names: Sequence[str] | None = None,
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")

        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.num_experts = num_experts

        if expert_names is None:
            expert_names = self.DEFAULT_EXPERT_NAMES[:num_experts]
        if len(expert_names) != num_experts:
            raise ValueError("expert_names length must match num_experts")
        self.expert_names = tuple(expert_names)

        input_dim = latent_dim + proprio_dim
        self.experts = nn.ModuleList(
            build_mlp(
                input_dim,
                hidden_dims=expert_hidden_dims,
                output_dim=action_dim,
                activation=activation,
            )
            for _ in range(num_experts)
        )
        self.router = build_mlp(
            latent_dim,
            hidden_dims=router_hidden_dims,
            output_dim=num_experts,
            activation=activation,
        )

    def _check_inputs(self, z: torch.Tensor, proprio: torch.Tensor) -> None:
        if z.dim() != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must be [B, {self.latent_dim}], got {tuple(z.shape)}")
        if proprio.dim() != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio must be [B, {self.proprio_dim}], got {tuple(proprio.shape)}")
        if z.shape[0] != proprio.shape[0]:
            raise ValueError(f"z and proprio must have the same batch size, got {z.shape[0]} and {proprio.shape[0]}")

    def forward(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_inputs(z, proprio)
        actor_input = torch.cat([z, proprio], dim=-1)
        router_logits = self.router(z)
        router_weights = torch.softmax(router_logits, dim=-1)
        expert_actions = torch.stack([expert(actor_input) for expert in self.experts], dim=1)

        action_mean = torch.sum(router_weights.unsqueeze(-1) * expert_actions, dim=1)
        return action_mean, router_weights, expert_actions, router_logits


class OrthogonalMoEActor(nn.Module):
    """Representation-level CTS-MoE actor with optional Gram-Schmidt features."""

    DEFAULT_EXPERT_NAMES = MoEActor.DEFAULT_EXPERT_NAMES

    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_experts: int = 4,
        expert_feature_dim: int = 128,
        expert_hidden_dims: Sequence[int] = (256, 128),
        router_hidden_dims: Sequence[int] = (128, 64),
        action_head_hidden_dims: Sequence[int] = (128,),
        expert_names: Sequence[str] | None = None,
        orthogonal_mode: str = "gram_schmidt",
        gate_activation: str = "softmax",
        use_expert_layernorm: bool = True,
        use_moe_output_layernorm: bool = True,
        gram_schmidt_eps: float = 1e-6,
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if expert_feature_dim < num_experts and orthogonal_mode == "gram_schmidt":
            raise ValueError("expert_feature_dim must be >= num_experts for Gram-Schmidt orthogonalization")
        if orthogonal_mode not in ("none", "gram_schmidt"):
            raise ValueError("orthogonal_mode must be 'none' or 'gram_schmidt'")
        if gate_activation not in ("softmax", "tanh", "sigmoid", "linear", "l2"):
            raise ValueError("gate_activation must be 'softmax', 'tanh', 'sigmoid', 'linear', or 'l2'")

        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.expert_feature_dim = expert_feature_dim
        self.orthogonal_mode = orthogonal_mode
        self.gate_activation = gate_activation
        self.gram_schmidt_eps = gram_schmidt_eps

        if expert_names is None:
            expert_names = self.DEFAULT_EXPERT_NAMES[:num_experts]
        if len(expert_names) != num_experts:
            raise ValueError("expert_names length must match num_experts")
        self.expert_names = tuple(expert_names)

        input_dim = latent_dim + proprio_dim
        self.experts = nn.ModuleList(
            build_mlp(
                input_dim,
                hidden_dims=expert_hidden_dims,
                output_dim=expert_feature_dim,
                activation=activation,
            )
            for _ in range(num_experts)
        )
        self.expert_norms = nn.ModuleList(
            nn.LayerNorm(expert_feature_dim) if use_expert_layernorm else nn.Identity()
            for _ in range(num_experts)
        )
        self.router = build_mlp(
            latent_dim,
            hidden_dims=router_hidden_dims,
            output_dim=num_experts,
            activation=activation,
        )
        self.moe_output_norm = nn.LayerNorm(expert_feature_dim) if use_moe_output_layernorm else nn.Identity()
        self.action_head = build_mlp(
            expert_feature_dim,
            hidden_dims=action_head_hidden_dims,
            output_dim=action_dim,
            activation=activation,
        )

    def _check_inputs(self, z: torch.Tensor, proprio: torch.Tensor) -> None:
        if z.dim() != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must be [B, {self.latent_dim}], got {tuple(z.shape)}")
        if proprio.dim() != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio must be [B, {self.proprio_dim}], got {tuple(proprio.shape)}")
        if z.shape[0] != proprio.shape[0]:
            raise ValueError(f"z and proprio must have the same batch size, got {z.shape[0]} and {proprio.shape[0]}")

    def apply_gate_activation(self, gate_logits: torch.Tensor) -> torch.Tensor:
        if self.gate_activation == "softmax":
            return torch.softmax(gate_logits, dim=-1)
        if self.gate_activation == "tanh":
            return torch.tanh(gate_logits)
        if self.gate_activation == "sigmoid":
            return torch.sigmoid(gate_logits)
        if self.gate_activation == "linear":
            return gate_logits
        if self.gate_activation == "l2":
            return gate_logits / (gate_logits.norm(dim=-1, keepdim=True) + self.gram_schmidt_eps)
        raise ValueError(f"Unknown gate_activation: {self.gate_activation}")

    def forward(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._check_inputs(z, proprio)
        actor_input = torch.cat([z, proprio], dim=-1)
        router_logits = self.router(z)
        gate_coeffs = self.apply_gate_activation(router_logits)

        expert_features_raw = torch.stack(
            [norm(expert(actor_input)) for expert, norm in zip(self.experts, self.expert_norms)],
            dim=1,
        )
        if self.orthogonal_mode == "gram_schmidt":
            expert_features_orth = batched_gram_schmidt(expert_features_raw, eps=self.gram_schmidt_eps)
        elif self.orthogonal_mode == "none":
            expert_features_orth = expert_features_raw
        else:
            raise ValueError(f"Unsupported orthogonal_mode: {self.orthogonal_mode}")

        mixed_feature = torch.sum(gate_coeffs.unsqueeze(-1) * expert_features_orth, dim=1)
        mixed_feature = self.moe_output_norm(mixed_feature)
        action_mean = self.action_head(mixed_feature)
        expert_actions = self.action_head(self.moe_output_norm(expert_features_orth))
        extras = {
            "gate_logits": router_logits,
            "gate_coeffs": gate_coeffs,
            "gate_weights": gate_coeffs,
            "gate_activation": self.gate_activation,
            "expert_features_raw": expert_features_raw,
            "expert_features_orth": expert_features_orth,
            "mixed_feature": mixed_feature,
        }
        # The second return value is kept as router_weights for compatibility;
        # it may not be a probability distribution when gate_activation != "softmax".
        return action_mean, gate_coeffs, expert_actions, router_logits, extras


class SparseMultiCritic(nn.Module):
    """Sparse task-conditioned critic with one value head per task."""

    TASK_NAMES = (
        "box_avoidance",
        "under_table",
        "stair_up",
        "flat",
    )

    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        num_tasks: int = 4,
        critic_hidden_dims: Sequence[int] = (256, 128),
        critic_shared_trunk: bool = False,
        trunk_hidden_dims: Sequence[int] | None = None,
        head_hidden_dims: Sequence[int] = (64,),
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        if num_tasks < 1:
            raise ValueError("num_tasks must be positive")

        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.num_tasks = num_tasks
        self.critic_shared_trunk = critic_shared_trunk
        input_dim = latent_dim + proprio_dim

        if critic_shared_trunk:
            trunk_hidden_dims = critic_hidden_dims if trunk_hidden_dims is None else trunk_hidden_dims
            trunk_output_dim = trunk_hidden_dims[-1] if trunk_hidden_dims else input_dim
            self.trunk = (
                build_mlp(input_dim, trunk_hidden_dims[:-1], trunk_output_dim, activation=activation)
                if trunk_hidden_dims
                else nn.Identity()
            )
            self.value_heads = nn.ModuleList(
                build_mlp(
                    trunk_output_dim,
                    hidden_dims=head_hidden_dims,
                    output_dim=1,
                    activation=activation,
                )
                for _ in range(num_tasks)
            )
        else:
            self.trunk = None
            self.value_heads = nn.ModuleList(
                build_mlp(
                    input_dim,
                    hidden_dims=critic_hidden_dims,
                    output_dim=1,
                    activation=activation,
                )
                for _ in range(num_tasks)
            )

    def _check_inputs(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor | None,
    ) -> None:
        if z.dim() != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must be [B, {self.latent_dim}], got {tuple(z.shape)}")
        if proprio.dim() != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio must be [B, {self.proprio_dim}], got {tuple(proprio.shape)}")
        if z.shape[0] != proprio.shape[0]:
            raise ValueError(f"z and proprio must have the same batch size, got {z.shape[0]} and {proprio.shape[0]}")
        if task_id is None:
            return
        if task_id.dim() != 1 or task_id.shape[0] != z.shape[0]:
            raise ValueError(f"task_id must be [B] with B={z.shape[0]}, got {tuple(task_id.shape)}")
        if torch.any(task_id < 0) or torch.any(task_id >= self.num_tasks):
            raise ValueError(f"task_id values must be in [0, {self.num_tasks - 1}]")

    def forward(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor | None = None,
        return_all_values: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._check_inputs(z, proprio, task_id)
        critic_input = torch.cat([z, proprio], dim=-1)
        if self.critic_shared_trunk:
            trunk_feature = self.trunk(critic_input)
            all_values = torch.stack([head(trunk_feature) for head in self.value_heads], dim=1)
        else:
            all_values = torch.stack([critic(critic_input) for critic in self.value_heads], dim=1)

        if task_id is None:
            if return_all_values:
                return all_values
            raise ValueError("task_id is required unless return_all_values=True")

        task_id = task_id.long().view(-1)
        batch_idx = torch.arange(z.shape[0], device=z.device)
        value = all_values[batch_idx, task_id]
        if return_all_values:
            return value, all_values
        return value


class StructureAwareCTSMoEPolicy(nn.Module):
    """End-to-end container for teacher/student CTS-MoE actor-critic modules."""

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        privileged_dim: int,
        latent_dim: int = 32,
        height_channels: int = 3,
        teacher_context_dim: int = 0,
        teacher_height_flat_dim: int | None = None,
        teacher_height_feature_dim: int = 128,
        teacher_privileged_feature_dim: int = 32,
        teacher_height_encoder_type: str = "mlp",
        teacher_height_hidden_dims: Sequence[int] = (512, 256),
        teacher_height_cnn_filters: Sequence[int] = (16, 32, 64),
        teacher_privileged_hidden_dims: Sequence[int] = (512, 256),
        student_perception_type: str = "grid",
        student_perception_dim: int | None = None,
        student_perception_channels: int = 1,
        student_proprio_feature_dim: int = 32,
        student_proprio_hidden_dims: Sequence[int] = (512, 256),
        student_depth_feature_dim: int = 128,
        student_depth_filters: Sequence[int] = (16, 32, 64),
        student_gru_hidden_dim: int = 256,
        student_gru_num_layers: int = 1,
        actor_type: str = "cts_moe",
        num_experts: int = 4,
        num_tasks: int = 4,
        expert_feature_dim: int = 128,
        expert_hidden_dims: Sequence[int] = (256, 128),
        router_hidden_dims: Sequence[int] = (128, 64),
        action_head_hidden_dims: Sequence[int] = (128,),
        expert_names: Sequence[str] | None = None,
        orthogonal_mode: str = "gram_schmidt",
        gate_activation: str = "softmax",
        use_expert_layernorm: bool = True,
        use_moe_output_layernorm: bool = True,
        gram_schmidt_eps: float = 1e-6,
        log_expert_metrics: bool = True,
        critic_hidden_dims: Sequence[int] = (256, 128),
        critic_shared_trunk: bool = False,
        critic_trunk_hidden_dims: Sequence[int] | None = None,
        critic_head_hidden_dims: Sequence[int] = (64,),
        init_log_std: float = 0.0,
        learnable_log_std: bool = True,
        semantic_decoupled_teacher: bool = True,
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        if actor_type not in ("cts_moe", "orthogonal_cts_moe"):
            raise ValueError("actor_type must be 'cts_moe' or 'orthogonal_cts_moe'")
        self.actor_type = actor_type
        self.log_expert_metrics = log_expert_metrics

        self.teacher_encoder = TeacherEncoder(
            privileged_dim=privileged_dim,
            latent_dim=latent_dim,
            height_channels=height_channels,
            num_tasks=num_tasks,
            context_dim=teacher_context_dim,
            height_flat_dim=teacher_height_flat_dim,
            semantic_decoupled=semantic_decoupled_teacher,
            height_feature_dim=teacher_height_feature_dim,
            privileged_feature_dim=teacher_privileged_feature_dim,
            height_encoder_type=teacher_height_encoder_type,
            height_hidden_dims=teacher_height_hidden_dims,
            height_cnn_filters=teacher_height_cnn_filters,
            privileged_hidden_dims=teacher_privileged_hidden_dims,
            activation=activation,
        )
        self.student_encoder = StudentEncoder(
            proprio_dim=proprio_dim,
            latent_dim=latent_dim,
            perception_type=student_perception_type,
            perception_dim=student_perception_dim,
            perception_channels=student_perception_channels,
            proprio_feature_dim=student_proprio_feature_dim,
            proprio_hidden_dims=student_proprio_hidden_dims,
            depth_feature_dim=student_depth_feature_dim,
            depth_filters=student_depth_filters,
            gru_hidden_dim=student_gru_hidden_dim,
            gru_num_layers=student_gru_num_layers,
            activation=activation,
        )
        if actor_type == "orthogonal_cts_moe":
            self.moe_actor = OrthogonalMoEActor(
                latent_dim=latent_dim,
                proprio_dim=proprio_dim,
                action_dim=action_dim,
                num_experts=num_experts,
                expert_feature_dim=expert_feature_dim,
                expert_hidden_dims=expert_hidden_dims,
                router_hidden_dims=router_hidden_dims,
                action_head_hidden_dims=action_head_hidden_dims,
                expert_names=expert_names,
                orthogonal_mode=orthogonal_mode,
                gate_activation=gate_activation,
                use_expert_layernorm=use_expert_layernorm,
                use_moe_output_layernorm=use_moe_output_layernorm,
                gram_schmidt_eps=gram_schmidt_eps,
                activation=activation,
            )
        else:
            self.moe_actor = MoEActor(
                latent_dim=latent_dim,
                proprio_dim=proprio_dim,
                action_dim=action_dim,
                num_experts=num_experts,
                expert_hidden_dims=expert_hidden_dims,
                router_hidden_dims=router_hidden_dims,
                expert_names=expert_names,
                activation=activation,
            )
        self.multi_critic = SparseMultiCritic(
            latent_dim=latent_dim,
            proprio_dim=proprio_dim,
            num_tasks=num_tasks,
            critic_hidden_dims=critic_hidden_dims,
            critic_shared_trunk=critic_shared_trunk,
            trunk_hidden_dims=critic_trunk_hidden_dims,
            head_hidden_dims=critic_head_hidden_dims,
            activation=activation,
        )

        log_std = torch.full((action_dim,), float(init_log_std))
        if learnable_log_std:
            self.log_std = nn.Parameter(log_std)
        else:
            self.register_buffer("log_std", log_std)

    @property
    def action_std(self) -> torch.Tensor:
        return torch.exp(self.log_std)

    def ppo_parameters(self):
        """Parameters for PPO updates, excluding the student encoder."""
        modules = [self.teacher_encoder, self.moe_actor, self.multi_critic]
        for module in modules:
            yield from module.parameters()
        if isinstance(self.log_std, nn.Parameter):
            yield self.log_std

    def student_parameters(self):
        """Parameters for student encoder distillation updates."""
        yield from self.student_encoder.parameters()

    def get_action_distribution(self, action_mean: torch.Tensor, action_std: torch.Tensor) -> Normal:
        return Normal(action_mean, action_std)

    def encode_teacher(
        self,
        height_scan: torch.Tensor,
        privileged_obs: torch.Tensor,
        return_task_id: bool = False,
        return_task_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.teacher_encoder(
            height_scan,
            privileged_obs,
            return_task_id=return_task_id,
            return_task_logits=return_task_logits,
        )

    def encode_student(self, proprio_history: torch.Tensor, perception: torch.Tensor) -> torch.Tensor:
        return self.student_encoder(proprio_history, perception)

    def act_teacher(
        self,
        height_scan: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_teacher = self.encode_teacher(height_scan, privileged_obs)
        return self.moe_actor(z_teacher, proprio)

    def act_student(
        self,
        proprio_history: torch.Tensor,
        perception: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_student = self.encode_student(proprio_history, perception)
        return self.moe_actor(z_student, proprio)

    def evaluate_teacher(
        self,
        height_scan: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        return_all_values: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z_teacher = self.encode_teacher(height_scan, privileged_obs)
        return self.multi_critic(z_teacher, proprio, task_id, return_all_values=return_all_values)

    def evaluate_student(
        self,
        proprio_history: torch.Tensor,
        perception: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        return_all_values: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z_student = self.encode_student(proprio_history, perception)
        return self.multi_critic(z_student, proprio, task_id, return_all_values=return_all_values)

    def distillation_loss(
        self,
        z_student: torch.Tensor,
        z_teacher: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(z_student, z_teacher.detach())

    def _check_forward_common(self, mode: str, proprio: torch.Tensor) -> None:
        if mode not in ("teacher", "student", "mixed"):
            raise ValueError("mode must be 'teacher', 'student', or 'mixed'")
        if proprio.dim() != 2:
            raise ValueError(f"proprio must be [B, proprio_dim], got {tuple(proprio.shape)}")

    def forward(
        self,
        *,
        mode: str,
        proprio: torch.Tensor,
        task_id: torch.Tensor | None = None,
        height_scan: torch.Tensor | None = None,
        privileged_obs: torch.Tensor | None = None,
        proprio_history: torch.Tensor | None = None,
        perception: torch.Tensor | None = None,
        student_mask: torch.Tensor | None = None,
        detach_student_in_mixed: bool = False,
        return_value: bool = False,
        return_all_values: bool = False,
    ) -> dict[str, Any]:
        """Run teacher, student, or mixed path and return a dict for PPO integration."""
        self._check_forward_common(mode, proprio)
        if mode == "teacher":
            if height_scan is None or privileged_obs is None:
                raise ValueError("teacher mode requires height_scan and privileged_obs")
            z, teacher_task_id, teacher_task_logits = self.encode_teacher(
                height_scan,
                privileged_obs,
                return_task_id=True,
                return_task_logits=True,
            )
            z_teacher = z
            z_student = None
        elif mode == "student":
            if proprio_history is None or perception is None:
                raise ValueError("student mode requires proprio_history and perception")
            z = self.encode_student(proprio_history, perception)
            z_teacher = None
            z_student = z
        elif mode == "mixed":
            if height_scan is None or privileged_obs is None:
                raise ValueError("mixed mode requires height_scan and privileged_obs")
            if proprio_history is None or perception is None:
                raise ValueError("mixed mode requires proprio_history and perception")
            if student_mask is None:
                raise ValueError("mixed mode requires student_mask")
            if student_mask.dtype != torch.bool or student_mask.dim() != 1 or student_mask.shape[0] != proprio.shape[0]:
                raise ValueError(
                    f"student_mask must be bool [B] with B={proprio.shape[0]}, got {tuple(student_mask.shape)}"
                )
            z_teacher, teacher_task_id, teacher_task_logits = self.encode_teacher(
                height_scan,
                privileged_obs,
                return_task_id=True,
                return_task_logits=True,
            )
            z_student = self.encode_student(proprio_history, perception)
            z_student_selected = z_student.detach() if detach_student_in_mixed else z_student
            z = torch.where(student_mask[:, None], z_student_selected, z_teacher)
        else:
            raise ValueError("mode must be 'teacher', 'student', or 'mixed'")

        actor_output = self.moe_actor(z, proprio)
        actor_extras = {}
        if len(actor_output) == 5:
            action_mean, router_weights, expert_actions, router_logits, actor_extras = actor_output
        else:
            action_mean, router_weights, expert_actions, router_logits = actor_output
        action_std = self.action_std.expand_as(action_mean)
        output = {
            "z": z,
            "action_mean": action_mean,
            "action_std": action_std,
            "distribution": self.get_action_distribution(action_mean, action_std),
            "router_weights": router_weights,
            "expert_actions": expert_actions,
            "router_logits": router_logits,
        }
        output.update(actor_extras)
        if z_teacher is not None:
            output["z_teacher"] = z_teacher
            output["teacher_task_id"] = teacher_task_id
            output["teacher_task_logits"] = teacher_task_logits
        if z_student is not None:
            output["z_student"] = z_student
        if student_mask is not None:
            output["student_mask"] = student_mask

        if return_value:
            if task_id is None:
                raise ValueError("task_id is required when return_value=True")
            critic_output = self.multi_critic(
                z,
                proprio,
                task_id,
                return_all_values=return_all_values,
            )
            if return_all_values:
                value, all_values = critic_output
                output["value"] = value
                output["all_values"] = all_values
            else:
                output["value"] = critic_output

        return output
