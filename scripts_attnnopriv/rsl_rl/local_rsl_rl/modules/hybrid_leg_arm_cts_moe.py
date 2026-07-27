"""Hybrid leg-arm CTS-MoE policy modules.

This module keeps the leg controller structure-aware and task-conditioned while
using a simple MLP actor-critic branch for the arm.  It is intentionally separate
from ``structure_aware_cts_moe.py`` so the original CTS-MoE baseline remains
unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .structure_aware_cts_moe import (
    OrthogonalMoEActor,
    SparseMultiCritic,
    StudentEncoder,
    TeacherEncoder,
    build_mlp,
)


def _make_index_tensor(indices: Sequence[int] | None) -> torch.Tensor | None:
    if indices is None:
        return None
    if len(indices) == 0:
        raise ValueError("Observation index lists must not be empty")
    return torch.tensor(list(indices), dtype=torch.long)


def _indexed_dim(indices: Sequence[int] | None, fallback_dim: int) -> int:
    return fallback_dim if indices is None else len(indices)


class ArmMLPActor(nn.Module):
    """Simple Gaussian-mean MLP actor for the arm action branch."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = build_mlp(
            obs_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"arm actor obs must be [B, {self.obs_dim}], got {tuple(obs.shape)}")
        return self.net(obs)


class ArmCritic(nn.Module):
    """Single-head MLP value function for arm-only rewards."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = build_mlp(
            obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"arm critic obs must be [B, {self.obs_dim}], got {tuple(obs.shape)}")
        return self.net(obs)


class HybridLegArmCTSMoEPolicy(nn.Module):
    """Teacher/student leg CTS-MoE policy with a decoupled arm MLP branch.

    The leg branch may observe the full proprioception vector by default,
    including arm joint position/velocity.  Index lists can be provided to narrow
    each branch's inputs without changing rollout storage.
    """

    def __init__(
        self,
        proprio_dim: int,
        privileged_dim: int,
        leg_action_dim: int = 12,
        arm_action_dim: int = 6,
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
        leg_critic_hidden_dims: Sequence[int] = (256, 128),
        leg_critic_shared_trunk: bool = False,
        leg_critic_trunk_hidden_dims: Sequence[int] | None = None,
        leg_critic_head_hidden_dims: Sequence[int] = (64,),
        arm_actor_hidden_dims: Sequence[int] = (256, 128),
        arm_critic_hidden_dims: Sequence[int] = (256, 128),
        leg_actor_proprio_indices: Sequence[int] | None = None,
        leg_critic_proprio_indices: Sequence[int] | None = None,
        arm_actor_obs_indices: Sequence[int] | None = None,
        arm_critic_obs_indices: Sequence[int] | None = None,
        arm_actor_use_latent: bool = False,
        arm_critic_use_latent: bool = True,
        arm_detach_latent: bool = True,
        init_leg_log_std: float = 0.0,
        init_arm_log_std: float = 0.0,
        learnable_log_std: bool = True,
        semantic_decoupled_teacher: bool = True,
        activation: str | type[nn.Module] | nn.Module = "elu",
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.privileged_dim = privileged_dim
        self.leg_action_dim = leg_action_dim
        self.arm_action_dim = arm_action_dim
        self.action_dim = leg_action_dim + arm_action_dim
        self.latent_dim = latent_dim
        self.log_expert_metrics = log_expert_metrics
        self.arm_actor_use_latent = arm_actor_use_latent
        self.arm_critic_use_latent = arm_critic_use_latent
        self.arm_detach_latent = arm_detach_latent

        self.register_buffer("leg_actor_proprio_indices", _make_index_tensor(leg_actor_proprio_indices), persistent=False)
        self.register_buffer("arm_actor_obs_indices", _make_index_tensor(arm_actor_obs_indices), persistent=False)

        leg_actor_proprio_dim = _indexed_dim(leg_actor_proprio_indices, proprio_dim)
        arm_actor_obs_dim = _indexed_dim(arm_actor_obs_indices, proprio_dim)

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
        self.leg_actor = OrthogonalMoEActor(
            latent_dim=latent_dim,
            proprio_dim=leg_actor_proprio_dim,
            action_dim=leg_action_dim,
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
        self.leg_critic = SparseMultiCritic(
            latent_dim=latent_dim,
            proprio_dim=privileged_dim,
            num_tasks=num_tasks,
            critic_hidden_dims=leg_critic_hidden_dims,
            critic_shared_trunk=leg_critic_shared_trunk,
            trunk_hidden_dims=leg_critic_trunk_hidden_dims,
            head_hidden_dims=leg_critic_head_hidden_dims,
            activation=activation,
        )

        arm_actor_input_dim = arm_actor_obs_dim + (latent_dim if arm_actor_use_latent else 0)
        arm_critic_input_dim = privileged_dim + (latent_dim if arm_critic_use_latent else 0)
        self.arm_actor = ArmMLPActor(
            obs_dim=arm_actor_input_dim,
            action_dim=arm_action_dim,
            hidden_dims=arm_actor_hidden_dims,
            activation=activation,
        )
        self.arm_critic = ArmCritic(
            obs_dim=arm_critic_input_dim,
            hidden_dims=arm_critic_hidden_dims,
            activation=activation,
        )

        leg_log_std = torch.full((leg_action_dim,), float(init_leg_log_std))
        arm_log_std = torch.full((arm_action_dim,), float(init_arm_log_std))
        if learnable_log_std:
            self.leg_log_std = nn.Parameter(leg_log_std)
            self.arm_log_std = nn.Parameter(arm_log_std)
        else:
            self.register_buffer("leg_log_std", leg_log_std)
            self.register_buffer("arm_log_std", arm_log_std)

    @property
    def leg_action_std(self) -> torch.Tensor:
        return torch.exp(self.leg_log_std)

    @property
    def arm_action_std(self) -> torch.Tensor:
        return torch.exp(self.arm_log_std)

    @property
    def action_std(self) -> torch.Tensor:
        return torch.cat([self.leg_action_std, self.arm_action_std], dim=-1)

    @property
    def moe_actor(self) -> OrthogonalMoEActor:
        """Compatibility alias for logging utilities that expect ``moe_actor``."""
        return self.leg_actor

    @property
    def multi_critic(self) -> SparseMultiCritic:
        """Compatibility alias for leg task critic."""
        return self.leg_critic

    def leg_parameters(self):
        yield from self.teacher_encoder.parameters()
        yield from self.leg_actor.parameters()
        yield from self.leg_critic.parameters()
        if isinstance(self.leg_log_std, nn.Parameter):
            yield self.leg_log_std

    def arm_parameters(self):
        yield from self.arm_actor.parameters()
        yield from self.arm_critic.parameters()
        if isinstance(self.arm_log_std, nn.Parameter):
            yield self.arm_log_std

    def student_parameters(self):
        yield from self.student_encoder.parameters()

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

    def distillation_loss(self, z_student: torch.Tensor, z_teacher: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(z_student, z_teacher.detach())

    def _select_proprio(self, proprio: torch.Tensor, indices: torch.Tensor | None) -> torch.Tensor:
        if proprio.dim() != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio must be [B, {self.proprio_dim}], got {tuple(proprio.shape)}")
        if indices is None:
            return proprio
        return proprio.index_select(dim=-1, index=indices)

    def _latent_for_arm(self, z: torch.Tensor) -> torch.Tensor:
        return z.detach() if self.arm_detach_latent else z

    def _build_arm_actor_obs(self, proprio: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        obs = self._select_proprio(proprio, self.arm_actor_obs_indices)
        if self.arm_actor_use_latent:
            obs = torch.cat([obs, self._latent_for_arm(z)], dim=-1)
        return obs

    def _check_privileged_obs(self, privileged_obs: torch.Tensor) -> None:
        if privileged_obs.dim() != 2 or privileged_obs.shape[-1] != self.privileged_dim:
            raise ValueError(f"privileged_obs must be [B, {self.privileged_dim}], got {tuple(privileged_obs.shape)}")

    def _build_arm_critic_obs(self, privileged_obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        self._check_privileged_obs(privileged_obs)
        obs = privileged_obs
        if self.arm_critic_use_latent:
            obs = torch.cat([obs, self._latent_for_arm(z)], dim=-1)
        return obs

    def _distribution(self, mean: torch.Tensor, std: torch.Tensor, name: str) -> Normal:
        if not torch.isfinite(mean).all():
            bad_count = (~torch.isfinite(mean)).sum().item()
            raise ValueError(f"{name} action mean contains {bad_count} non-finite values")
        if not torch.isfinite(std).all():
            bad_count = (~torch.isfinite(std)).sum().item()
            raise ValueError(f"{name} action std contains {bad_count} non-finite values")
        return Normal(mean, std)

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
        return_leg_value: bool = False,
        return_arm_value: bool = False,
        return_value: bool = False,
        return_all_leg_values: bool = False,
    ) -> dict[str, Any]:
        if mode not in ("teacher", "student", "mixed"):
            raise ValueError("mode must be 'teacher', 'student', or 'mixed'")

        z_teacher = None
        z_student = None
        teacher_task_id = None
        teacher_task_logits = None
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
        elif mode == "student":
            if proprio_history is None or perception is None:
                raise ValueError("student mode requires proprio_history and perception")
            z = self.encode_student(proprio_history, perception)
            z_student = z
        else:
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

        leg_proprio = self._select_proprio(proprio, self.leg_actor_proprio_indices)
        leg_output = self.leg_actor(z, leg_proprio)
        leg_action_mean, leg_router_weights, leg_expert_actions, leg_router_logits, leg_extras = leg_output
        arm_obs = self._build_arm_actor_obs(proprio, z)
        arm_action_mean = self.arm_actor(arm_obs)

        leg_action_std = self.leg_action_std.expand_as(leg_action_mean)
        arm_action_std = self.arm_action_std.expand_as(arm_action_mean)
        action_mean = torch.cat([leg_action_mean, arm_action_mean], dim=-1)
        action_std = torch.cat([leg_action_std, arm_action_std], dim=-1)

        output: dict[str, Any] = {
            "z": z,
            "action_mean": action_mean,
            "action_std": action_std,
            "distribution": self._distribution(action_mean, action_std, "full"),
            "leg_action_mean": leg_action_mean,
            "leg_action_std": leg_action_std,
            "leg_distribution": self._distribution(leg_action_mean, leg_action_std, "leg"),
            "arm_action_mean": arm_action_mean,
            "arm_action_std": arm_action_std,
            "arm_distribution": self._distribution(arm_action_mean, arm_action_std, "arm"),
            "router_weights": leg_router_weights,
            "expert_actions": leg_expert_actions,
            "router_logits": leg_router_logits,
        }
        output.update(leg_extras)
        if z_teacher is not None:
            output["z_teacher"] = z_teacher
            output["teacher_task_id"] = teacher_task_id
            output["teacher_task_logits"] = teacher_task_logits
        if z_student is not None:
            output["z_student"] = z_student
        if student_mask is not None:
            output["student_mask"] = student_mask

        return_leg_value = return_leg_value or return_value
        return_arm_value = return_arm_value or return_value
        if return_leg_value:
            if task_id is None:
                raise ValueError("task_id is required when return_leg_value=True")
            if privileged_obs is None:
                raise ValueError("privileged_obs is required when return_leg_value=True")
            self._check_privileged_obs(privileged_obs)
            leg_value = self.leg_critic(
                z,
                privileged_obs,
                task_id,
                return_all_values=return_all_leg_values,
            )
            if return_all_leg_values:
                leg_value, all_leg_values = leg_value
                output["all_leg_values"] = all_leg_values
            output["leg_value"] = leg_value
            output["value"] = leg_value
        if return_arm_value:
            if privileged_obs is None:
                raise ValueError("privileged_obs is required when return_arm_value=True")
            arm_critic_obs = self._build_arm_critic_obs(privileged_obs, z)
            output["arm_value"] = self.arm_critic(arm_critic_obs)

        return output
