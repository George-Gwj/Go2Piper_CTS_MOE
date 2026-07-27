from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from local_rsl_rl.modules import compute_orthogonality_metrics
from local_rsl_rl.storage import HybridLegArmRolloutStorage


class HybridLegArmPPO:
    """PPO for HybridLegArmCTSMoEPolicy.

    Rollouts are collected jointly, but PPO updates are decoupled:
    - leg rewards/advantages update teacher encoder, leg CTS-MoE actor, leg critic, leg log-std;
    - arm rewards/advantages update only arm MLP actor, arm critic, arm log-std;
    - mixed mode optionally distills the student latent from the teacher latent.
    """

    VALID_TRAINING_MODES = ("teacher", "mixed")

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        leg_value_loss_coef: float = 1.0,
        arm_value_loss_coef: float = 1.0,
        leg_entropy_coef: float = 0.0,
        arm_entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        leg_learning_rate: float | None = None,
        arm_learning_rate: float | None = None,
        student_learning_rate: float | None = None,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = 0.01,
        device: str = "cpu",
        eps: float = 1e-5,
        training_mode: str = "mixed",
        distillation_loss_coef: float = 1.0,
        student_rollout_ratio: float = 0.15,
        router_entropy_coef: float = 0.0,
        router_balance_coef: float = 0.0,
        router_logit_l2_coef: float = 0.0,
        lambda_orth: float = 0.0,
        orth_loss_on: str = "raw",
        per_task_leg_advantage_normalization: bool = True,
        normalize_arm_advantage: bool = True,
    ):
        self.device = device
        self.policy = policy.to(device)
        self.storage = None
        self.transition = HybridLegArmRolloutStorage.Transition()

        self.leg_learning_rate = learning_rate if leg_learning_rate is None else leg_learning_rate
        self.arm_learning_rate = learning_rate if arm_learning_rate is None else arm_learning_rate
        self.student_learning_rate = learning_rate if student_learning_rate is None else student_learning_rate
        self.leg_optimizer = optim.Adam(self.policy.leg_parameters(), lr=self.leg_learning_rate, eps=eps)
        self.arm_optimizer = optim.Adam(self.policy.arm_parameters(), lr=self.arm_learning_rate, eps=eps)
        self.student_optimizer = optim.Adam(self.policy.student_parameters(), lr=self.student_learning_rate, eps=eps)

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.leg_value_loss_coef = leg_value_loss_coef
        self.arm_value_loss_coef = arm_value_loss_coef
        self.leg_entropy_coef = leg_entropy_coef
        self.arm_entropy_coef = arm_entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        if training_mode not in self.VALID_TRAINING_MODES:
            raise ValueError(f"training_mode must be one of {self.VALID_TRAINING_MODES}, got {training_mode!r}")
        self.training_mode = training_mode
        self.distillation_loss_coef = distillation_loss_coef
        if student_rollout_ratio < 0.0 or student_rollout_ratio > 1.0:
            raise ValueError("student_rollout_ratio must be in [0, 1]")
        self.student_rollout_ratio = student_rollout_ratio
        self.router_entropy_coef = router_entropy_coef
        self.router_balance_coef = router_balance_coef
        self.router_logit_l2_coef = router_logit_l2_coef
        self.lambda_orth = lambda_orth
        if orth_loss_on not in ("raw", "orth"):
            raise ValueError("orth_loss_on must be 'raw' or 'orth'")
        self.orth_loss_on = orth_loss_on
        self.per_task_leg_advantage_normalization = per_task_leg_advantage_normalization
        self.normalize_arm_advantage = normalize_arm_advantage

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        proprio_shape,
        height_scan_shape,
        privileged_obs_shape,
        proprio_history_shape,
        perception_shape,
        leg_actions_shape,
        arm_actions_shape,
    ):
        self.storage = HybridLegArmRolloutStorage(
            num_envs,
            num_transitions_per_env,
            proprio_shape,
            height_scan_shape,
            privileged_obs_shape,
            proprio_history_shape,
            perception_shape,
            leg_actions_shape,
            arm_actions_shape,
            self.policy.leg_actor.num_experts,
            self.device,
        )

    def act(
        self,
        proprio: torch.Tensor,
        height_scan: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
        perception: torch.Tensor,
        task_id: torch.Tensor,
        student_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        student_mask = self._resolve_student_mask(proprio.shape[0], proprio.device, student_mask)
        forward_kwargs = self._build_forward_kwargs(
            proprio=proprio,
            task_id=task_id,
            height_scan=height_scan,
            privileged_obs=privileged_obs,
            proprio_history=proprio_history,
            perception=perception,
            student_mask=student_mask,
            return_leg_value=True,
            return_arm_value=True,
        )
        with torch.no_grad():
            out = self.policy(**forward_kwargs)
            leg_actions = out["leg_distribution"].sample()
            arm_actions = out["arm_distribution"].sample()
            actions = torch.cat([leg_actions, arm_actions], dim=-1)
            leg_actions_log_prob = out["leg_distribution"].log_prob(leg_actions).sum(dim=-1)
            arm_actions_log_prob = out["arm_distribution"].log_prob(arm_actions).sum(dim=-1)

        self.transition.proprio = proprio
        self.transition.height_scan = height_scan
        self.transition.privileged_obs = privileged_obs
        self.transition.proprio_history = proprio_history
        self.transition.perception = perception
        self.transition.task_id = task_id
        self.transition.student_mask = student_mask
        self.transition.actions = actions.detach()
        self.transition.leg_actions = leg_actions.detach()
        self.transition.arm_actions = arm_actions.detach()
        self.transition.leg_values = out["leg_value"].detach()
        self.transition.arm_values = out["arm_value"].detach()
        self.transition.leg_actions_log_prob = leg_actions_log_prob.detach()
        self.transition.arm_actions_log_prob = arm_actions_log_prob.detach()
        self.transition.leg_action_mean = out["leg_action_mean"].detach()
        self.transition.leg_action_sigma = out["leg_action_std"].detach()
        self.transition.arm_action_mean = out["arm_action_mean"].detach()
        self.transition.arm_action_sigma = out["arm_action_std"].detach()
        self.transition.router_weights = out["router_weights"].detach()
        return actions.detach()

    def process_env_step(
        self,
        leg_rewards: torch.Tensor,
        arm_rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
    ):
        self.transition.leg_rewards = leg_rewards.clone()
        self.transition.arm_rewards = arm_rewards.clone()
        self.transition.dones = dones

        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            leg_bootstrap = self.gamma * torch.squeeze(self.transition.leg_values * time_outs, 1)
            arm_bootstrap = self.gamma * torch.squeeze(self.transition.arm_values * time_outs, 1)
            self.transition.leg_rewards += leg_bootstrap.view_as(self.transition.leg_rewards)
            self.transition.arm_rewards += arm_bootstrap.view_as(self.transition.arm_rewards)

        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(
        self,
        proprio: torch.Tensor,
        height_scan: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
        perception: torch.Tensor,
        task_id: torch.Tensor,
        student_mask: torch.Tensor | None = None,
    ):
        student_mask = self._resolve_student_mask(proprio.shape[0], proprio.device, student_mask)
        forward_kwargs = self._build_forward_kwargs(
            proprio=proprio,
            task_id=task_id,
            height_scan=height_scan,
            privileged_obs=privileged_obs,
            proprio_history=proprio_history,
            perception=perception,
            student_mask=student_mask,
            return_leg_value=True,
            return_arm_value=True,
        )
        with torch.no_grad():
            out = self.policy(**forward_kwargs)
        self.storage.compute_returns(
            out["leg_value"].detach(),
            out["arm_value"].detach(),
            self.gamma,
            self.lam,
            normalize_advantage=True,
            per_task_leg_advantage_normalization=self.per_task_leg_advantage_normalization,
            normalize_arm_advantage=self.normalize_arm_advantage,
        )

    def update(self):
        leg_stats = self._update_leg()
        arm_stats = self._update_arm()
        distill_stats = self._update_student_distillation()

        loss_dict = {}
        loss_dict.update({f"leg/{key}": value for key, value in leg_stats.items()})
        loss_dict.update({f"arm/{key}": value for key, value in arm_stats.items()})
        loss_dict.update(distill_stats)
        loss_dict.update(self._compute_per_task_router_weight_stats())
        loss_dict["student_rollout_ratio"] = self.storage.student_masks.float().mean().item()
        self.storage.clear()
        return loss_dict

    def _update_leg(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_router_entropy = 0.0
        mean_router_balance_loss = 0.0
        mean_router_logit_l2_loss = 0.0
        mean_orth_loss = 0.0
        skipped_nonfinite_updates = 0
        orth_metric_sums: dict[str, float] = {}
        num_orth_metric_updates = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            (
                proprio_batch,
                height_scan_batch,
                privileged_obs_batch,
                proprio_history_batch,
                perception_batch,
                task_id_batch,
                student_mask_batch,
                leg_actions_batch,
                _arm_actions_batch,
                target_leg_values_batch,
                _target_arm_values_batch,
                leg_advantages_batch,
                _arm_advantages_batch,
                leg_returns_batch,
                _arm_returns_batch,
                old_leg_actions_log_prob_batch,
                _old_arm_actions_log_prob_batch,
                old_leg_mu_batch,
                old_leg_sigma_batch,
                _old_arm_mu_batch,
                _old_arm_sigma_batch,
            ) = self._sanitize_update_batch(*batch)

            forward_kwargs = self._build_forward_kwargs(
                proprio=proprio_batch,
                task_id=task_id_batch,
                height_scan=height_scan_batch,
                privileged_obs=privileged_obs_batch,
                proprio_history=proprio_history_batch,
                perception=perception_batch,
                student_mask=student_mask_batch,
                return_leg_value=True,
                return_arm_value=False,
            )
            try:
                out = self.policy(**forward_kwargs)
            except ValueError as exc:
                if "non-finite" not in str(exc):
                    raise
                skipped_nonfinite_updates += 1
                continue

            dist = out["leg_distribution"]
            actions_log_prob_batch = dist.log_prob(leg_actions_batch).sum(dim=-1)
            entropy_batch = dist.entropy().sum(dim=-1)
            value_batch = out["leg_value"]
            self._update_learning_rate(
                old_leg_mu_batch,
                old_leg_sigma_batch,
                out["leg_action_mean"],
                out["leg_action_std"],
                self.leg_optimizer,
                "leg_learning_rate",
            )

            surrogate_loss = self._ppo_surrogate_loss(
                actions_log_prob_batch,
                old_leg_actions_log_prob_batch,
                leg_advantages_batch,
            )
            value_loss = self._compute_value_loss(value_batch, target_leg_values_batch, leg_returns_batch)
            router_aux = self._router_auxiliary_loss(
                out["router_weights"],
                out["router_logits"],
                gate_activation=out.get("gate_activation", getattr(self.policy.leg_actor, "gate_activation", "softmax")),
            )
            orth_loss = self._orthogonality_loss(out)
            leg_loss = (
                surrogate_loss
                + self.leg_value_loss_coef * value_loss
                - self.leg_entropy_coef * entropy_batch.mean()
                + router_aux["loss"]
                + self.lambda_orth * orth_loss
            )
            if not torch.isfinite(leg_loss):
                skipped_nonfinite_updates += 1
                continue

            self.leg_optimizer.zero_grad()
            leg_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(list(self.policy.leg_parameters()), self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                self.leg_optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite_updates += 1
                continue
            self.leg_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_router_entropy += router_aux["entropy"].item()
            mean_router_balance_loss += router_aux["balance"].item()
            mean_router_logit_l2_loss += router_aux["logit_l2"].item()
            mean_orth_loss += orth_loss.item()
            batch_orth_metrics = self._orthogonality_metrics(out)
            if batch_orth_metrics:
                for key, value in batch_orth_metrics.items():
                    orth_metric_sums[key] = orth_metric_sums.get(key, 0.0) + value.item()
                num_orth_metric_updates += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        stats = {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "router_entropy": mean_router_entropy / num_updates,
            "router_balance": mean_router_balance_loss / num_updates,
            "router_logit_l2": mean_router_logit_l2_loss / num_updates,
            "orth_loss": mean_orth_loss / num_updates,
            "skipped_nonfinite_updates": skipped_nonfinite_updates,
            "learning_rate": self.leg_learning_rate,
        }
        if num_orth_metric_updates > 0:
            stats.update({key: value / num_orth_metric_updates for key, value in orth_metric_sums.items()})
        return stats

    def _update_arm(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        skipped_nonfinite_updates = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            (
                proprio_batch,
                height_scan_batch,
                privileged_obs_batch,
                proprio_history_batch,
                perception_batch,
                task_id_batch,
                student_mask_batch,
                _leg_actions_batch,
                arm_actions_batch,
                _target_leg_values_batch,
                target_arm_values_batch,
                _leg_advantages_batch,
                arm_advantages_batch,
                _leg_returns_batch,
                arm_returns_batch,
                _old_leg_actions_log_prob_batch,
                old_arm_actions_log_prob_batch,
                _old_leg_mu_batch,
                _old_leg_sigma_batch,
                old_arm_mu_batch,
                old_arm_sigma_batch,
            ) = self._sanitize_update_batch(*batch)

            forward_kwargs = self._build_forward_kwargs(
                proprio=proprio_batch,
                task_id=task_id_batch,
                height_scan=height_scan_batch,
                privileged_obs=privileged_obs_batch,
                proprio_history=proprio_history_batch,
                perception=perception_batch,
                student_mask=student_mask_batch,
                return_leg_value=False,
                return_arm_value=True,
            )
            try:
                out = self.policy(**forward_kwargs)
            except ValueError as exc:
                if "non-finite" not in str(exc):
                    raise
                skipped_nonfinite_updates += 1
                continue

            dist = out["arm_distribution"]
            actions_log_prob_batch = dist.log_prob(arm_actions_batch).sum(dim=-1)
            entropy_batch = dist.entropy().sum(dim=-1)
            value_batch = out["arm_value"]
            self._update_learning_rate(
                old_arm_mu_batch,
                old_arm_sigma_batch,
                out["arm_action_mean"],
                out["arm_action_std"],
                self.arm_optimizer,
                "arm_learning_rate",
            )

            surrogate_loss = self._ppo_surrogate_loss(
                actions_log_prob_batch,
                old_arm_actions_log_prob_batch,
                arm_advantages_batch,
            )
            value_loss = self._compute_value_loss(value_batch, target_arm_values_batch, arm_returns_batch)
            arm_loss = (
                surrogate_loss
                + self.arm_value_loss_coef * value_loss
                - self.arm_entropy_coef * entropy_batch.mean()
            )
            if not torch.isfinite(arm_loss):
                skipped_nonfinite_updates += 1
                continue

            self.arm_optimizer.zero_grad()
            arm_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(list(self.policy.arm_parameters()), self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                self.arm_optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite_updates += 1
                continue
            self.arm_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "skipped_nonfinite_updates": skipped_nonfinite_updates,
            "learning_rate": self.arm_learning_rate,
        }

    def _update_student_distillation(self) -> dict[str, float]:
        mean_distillation_loss = 0.0
        num_distill_updates = 0
        if self.training_mode != "mixed":
            return {"distillation": 0.0}

        generator = self.storage.mini_batch_generator(self.num_mini_batches, 1)
        for batch in generator:
            (
                _proprio_batch,
                height_scan_batch,
                privileged_obs_batch,
                proprio_history_batch,
                perception_batch,
                _task_id_batch,
                student_mask_batch,
                *_rest,
            ) = batch
            if not student_mask_batch.any():
                continue

            mask = student_mask_batch
            with torch.no_grad():
                z_teacher = self.policy.encode_teacher(height_scan_batch[mask], privileged_obs_batch[mask])
            z_student = self.policy.encode_student(proprio_history_batch[mask], perception_batch[mask])
            distillation_loss = self.policy.distillation_loss(z_student, z_teacher)
            student_loss = self.distillation_loss_coef * distillation_loss

            self.student_optimizer.zero_grad()
            student_loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy.student_parameters()), self.max_grad_norm)
            self.student_optimizer.step()

            mean_distillation_loss += distillation_loss.item()
            num_distill_updates += 1

        return {"distillation": mean_distillation_loss / max(num_distill_updates, 1)}

    def _resolve_student_mask(
        self,
        num_envs: int,
        device: torch.device | str,
        student_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training_mode == "teacher":
            return torch.zeros(num_envs, dtype=torch.bool, device=device)

        if student_mask is not None:
            if student_mask.dim() != 1 or student_mask.shape[0] != num_envs:
                raise ValueError(f"student_mask must be [B] with B={num_envs}, got {tuple(student_mask.shape)}")
            return student_mask.to(device=device, dtype=torch.bool)

        mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        num_student_envs = int(round(num_envs * self.student_rollout_ratio))
        if num_student_envs <= 0:
            return mask
        if num_student_envs >= num_envs:
            return torch.ones(num_envs, dtype=torch.bool, device=device)

        student_indices = torch.randperm(num_envs, device=device)[:num_student_envs]
        mask[student_indices] = True
        return mask

    def _build_forward_kwargs(
        self,
        *,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        height_scan: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
        perception: torch.Tensor,
        student_mask: torch.Tensor,
        return_leg_value: bool,
        return_arm_value: bool,
    ) -> dict:
        if self.training_mode == "teacher":
            return {
                "mode": "teacher",
                "proprio": proprio,
                "task_id": task_id,
                "height_scan": height_scan,
                "privileged_obs": privileged_obs,
                "return_leg_value": return_leg_value,
                "return_arm_value": return_arm_value,
            }
        return {
            "mode": "mixed",
            "proprio": proprio,
            "task_id": task_id,
            "height_scan": height_scan,
            "privileged_obs": privileged_obs,
            "proprio_history": proprio_history,
            "perception": perception,
            "student_mask": student_mask,
            "detach_student_in_mixed": True,
            "return_leg_value": return_leg_value,
            "return_arm_value": return_arm_value,
        }

    @staticmethod
    def _finite(tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(tensor):
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def _sanitize_update_batch(self, *batches: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self._finite(batch) for batch in batches)

    def _ppo_surrogate_loss(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
        surrogate = -advantages_batch.squeeze(-1) * ratio
        surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
            ratio,
            1.0 - self.clip_param,
            1.0 + self.clip_param,
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _compute_value_loss(
        self,
        value_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
        returns_batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param,
                self.clip_param,
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            return torch.max(value_losses, value_losses_clipped).mean()
        return (returns_batch - value_batch).pow(2).mean()

    def _update_learning_rate(
        self,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
        mu_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
        optimizer: optim.Optimizer,
        lr_attr_name: str,
    ):
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                dim=-1,
            )
            kl_mean = torch.mean(kl)
            if not torch.isfinite(kl_mean):
                return
            learning_rate = getattr(self, lr_attr_name)
            if kl_mean > self.desired_kl * 2.0:
                learning_rate = max(1e-5, learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                learning_rate = min(1e-2, learning_rate * 1.5)
            setattr(self, lr_attr_name, learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

    def _router_auxiliary_loss(
        self,
        router_weights: torch.Tensor,
        router_logits: torch.Tensor,
        gate_activation: str = "softmax",
    ) -> dict[str, torch.Tensor]:
        if gate_activation == "softmax":
            router_entropy = -(router_weights * torch.log(router_weights + 1e-8)).sum(dim=-1).mean()
            mean_weights = router_weights.mean(dim=0)
            uniform = torch.full_like(mean_weights, 1.0 / mean_weights.numel())
            router_balance_loss = (mean_weights - uniform).pow(2).mean()
        else:
            router_entropy = router_logits.new_zeros(())
            router_balance_loss = router_logits.new_zeros(())
        router_logit_l2_loss = router_logits.pow(2).mean()
        loss = (
            -self.router_entropy_coef * router_entropy
            + self.router_balance_coef * router_balance_loss
            + self.router_logit_l2_coef * router_logit_l2_loss
        )
        return {
            "loss": loss,
            "entropy": router_entropy.detach(),
            "balance": router_balance_loss.detach(),
            "logit_l2": router_logit_l2_loss.detach(),
        }

    def _orthogonality_loss(self, out: dict) -> torch.Tensor:
        reference = out.get("expert_features_raw" if self.orth_loss_on == "raw" else "expert_features_orth")
        if reference is None or self.lambda_orth == 0.0:
            return out["leg_action_mean"].new_zeros(())

        features = F.normalize(reference, dim=-1, eps=1e-6)
        gram = torch.matmul(features, features.transpose(-1, -2))
        num_experts = gram.shape[-1]
        eye = torch.eye(num_experts, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        off_diag = gram - eye
        return off_diag.pow(2).mean()

    def _orthogonality_metrics(self, out: dict) -> dict[str, torch.Tensor]:
        if not getattr(self.policy, "log_expert_metrics", False):
            return {}
        features = out.get("expert_features_orth")
        if features is None:
            return {}
        return compute_orthogonality_metrics(
            features,
            gate_weights=out.get("router_weights"),
            gate_activation=out.get("gate_activation", getattr(self.policy.leg_actor, "gate_activation", "softmax")),
            eps=1e-6,
        )

    def _compute_per_task_router_weight_stats(self) -> dict[str, float]:
        flat_weights = self.storage.router_weights.flatten(0, 1)
        flat_task_ids = self.storage.task_ids.flatten(0, 1)
        task_names = self.policy.leg_critic.TASK_NAMES[: self.policy.leg_critic.num_tasks]
        expert_names = self.policy.leg_actor.expert_names

        stats: dict[str, float] = {}
        for task_id, task_name in enumerate(task_names):
            mask = flat_task_ids == task_id
            if not mask.any():
                continue
            task_weights = flat_weights[mask].mean(dim=0)
            for expert_idx, expert_name in enumerate(expert_names):
                stats[f"Router/{task_name}/{expert_name}"] = task_weights[expert_idx].item()
        return stats
