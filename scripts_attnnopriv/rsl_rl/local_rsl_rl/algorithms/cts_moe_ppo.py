from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from local_rsl_rl.storage import CTSMoERolloutStorage
from local_rsl_rl.utils import PerTaskPopArt


class CTSMoEPPO:
    """PPO for StructureAwareCTSMoEPolicy.

    PPO gradients update teacher_encoder, moe_actor, multi_critic, and log_std.
    The student encoder is updated only through latent distillation.
    """

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        student_learning_rate: float | None = None,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = 0.01,
        device: str = "cpu",
        eps: float = 1e-5,
        distillation_loss_coef: float = 1.0,
        student_rollout_ratio: float = 0.15,
        router_entropy_coef: float = 0.0,
        router_balance_coef: float = 0.0,
        router_logit_l2_coef: float = 0.0,
        per_task_advantage_normalization: bool = True,
        use_popart: bool = False,
        popart_beta: float = 0.99999,
        popart_eps: float = 1e-5,
        popart_min_std: float = 1e-2,
        popart_use_output_rescale: bool = True,
        popart_value_loss: str = "huber",
        popart_huber_delta: float = 1.0,
        value_loss_per_task_average: bool = True,
    ):
        self.device = device
        self.policy = policy.to(device)
        self.storage = None
        self.transition = CTSMoERolloutStorage.Transition()

        self.optimizer = optim.Adam(self.policy.ppo_parameters(), lr=learning_rate, eps=eps)
        self.student_optimizer = optim.Adam(
            self.policy.student_parameters(),
            lr=learning_rate if student_learning_rate is None else student_learning_rate,
            eps=eps,
        )

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.distillation_loss_coef = distillation_loss_coef
        if student_rollout_ratio < 0.0 or student_rollout_ratio > 1.0:
            raise ValueError("student_rollout_ratio must be in [0, 1]")
        self.student_rollout_ratio = student_rollout_ratio
        self.router_entropy_coef = router_entropy_coef
        self.router_balance_coef = router_balance_coef
        self.router_logit_l2_coef = router_logit_l2_coef
        self.per_task_advantage_normalization = per_task_advantage_normalization
        self.use_popart = use_popart
        self.popart_use_output_rescale = popart_use_output_rescale
        self.popart_value_loss = popart_value_loss
        self.popart_huber_delta = popart_huber_delta
        self.value_loss_per_task_average = value_loss_per_task_average
        if popart_value_loss not in ("huber", "mse"):
            raise ValueError("popart_value_loss must be 'huber' or 'mse'")
        self.popart = (
            PerTaskPopArt(
                num_tasks=self.policy.multi_critic.num_tasks,
                beta=popart_beta,
                eps=popart_eps,
                min_std=popart_min_std,
                device=device,
            )
            if use_popart
            else None
        )

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        proprio_shape,
        height_scan_shape,
        privileged_obs_shape,
        proprio_history_shape,
        perception_shape,
        actions_shape,
    ):
        self.storage = CTSMoERolloutStorage(
            num_envs,
            num_transitions_per_env,
            proprio_shape,
            height_scan_shape,
            privileged_obs_shape,
            proprio_history_shape,
            perception_shape,
            actions_shape,
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
        with torch.no_grad():
            out = self.policy(
                mode="mixed",
                proprio=proprio,
                task_id=task_id,
                height_scan=height_scan,
                privileged_obs=privileged_obs,
                proprio_history=proprio_history,
                perception=perception,
                student_mask=student_mask,
                detach_student_in_mixed=True,
                return_value=True,
            )
            actions = out["distribution"].sample()
            actions_log_prob = out["distribution"].log_prob(actions).sum(dim=-1)
            value = out["value"]
            value_raw = self.popart.denormalize(task_id, value) if self.use_popart else value

        self.transition.proprio = proprio
        self.transition.height_scan = height_scan
        self.transition.privileged_obs = privileged_obs
        self.transition.proprio_history = proprio_history
        self.transition.perception = perception
        self.transition.task_id = task_id
        self.transition.student_mask = student_mask
        self.transition.actions = actions.detach()
        self.transition.values = value_raw.detach()
        self.transition.actions_log_prob = actions_log_prob.detach()
        self.transition.action_mean = out["action_mean"].detach()
        self.transition.action_sigma = out["action_std"].detach()
        return actions.detach()

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

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
        with torch.no_grad():
            out = self.policy(
                mode="mixed",
                proprio=proprio,
                task_id=task_id,
                height_scan=height_scan,
                privileged_obs=privileged_obs,
                proprio_history=proprio_history,
                perception=perception,
                student_mask=student_mask,
                detach_student_in_mixed=True,
                return_value=True,
            )
            value = out["value"]
            last_values = self.popart.denormalize(task_id, value) if self.use_popart else value
            last_values = last_values.detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=True,
            per_task_advantage_normalization=self.per_task_advantage_normalization,
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_router_entropy = 0.0
        mean_router_balance_loss = 0.0
        mean_router_logit_l2_loss = 0.0
        mean_returns_norm_mean = 0.0
        mean_returns_norm_std = 0.0

        if self.use_popart:
            flat_task_ids = self.storage.task_ids.flatten(0, 1)
            flat_returns = self.storage.returns.flatten(0, 1)
            old_stats, new_stats = self.popart.update(flat_task_ids, flat_returns)
            if self.popart_use_output_rescale:
                self._apply_popart_rescale_to_critic_heads(old_stats, new_stats)

        # Phase 1: PPO update. Student latent is detached in mixed mode, so PPO
        # gradients update only teacher_encoder, moe_actor, multi_critic, log_std.
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            proprio_batch,
            height_scan_batch,
            privileged_obs_batch,
            proprio_history_batch,
            perception_batch,
            task_id_batch,
            student_mask_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            out = self.policy(
                mode="mixed",
                proprio=proprio_batch,
                task_id=task_id_batch,
                height_scan=height_scan_batch,
                privileged_obs=privileged_obs_batch,
                proprio_history=proprio_history_batch,
                perception=perception_batch,
                student_mask=student_mask_batch,
                detach_student_in_mixed=True,
                return_value=True,
            )
            dist = out["distribution"]
            actions_log_prob_batch = dist.log_prob(actions_batch).sum(dim=-1)
            entropy_batch = dist.entropy().sum(dim=-1)
            value_batch = out["value"]

            self._update_learning_rate(old_mu_batch, old_sigma_batch, out["action_mean"], out["action_std"])

            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            value_loss, returns_norm_stats = self._compute_value_loss(
                value_batch,
                target_values_batch,
                returns_batch,
                task_id_batch,
            )

            router_aux = self._router_auxiliary_loss(out["router_weights"], out["router_logits"])

            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + router_aux["loss"]
            )

            self.optimizer.zero_grad()
            ppo_loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy.ppo_parameters()), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_router_entropy += router_aux["entropy"].item()
            mean_router_balance_loss += router_aux["balance"].item()
            mean_router_logit_l2_loss += router_aux["logit_l2"].item()
            mean_returns_norm_mean += returns_norm_stats["mean"].item()
            mean_returns_norm_std += returns_norm_stats["std"].item()

        num_updates = self.num_learning_epochs * self.num_mini_batches

        # Phase 2: student distillation after PPO update. Only trajectories that
        # actually used student latent during rollout/update are used here.
        mean_distillation_loss = 0.0
        num_distill_updates = 0
        distill_generator = self.storage.mini_batch_generator(self.num_mini_batches, 1)
        for (
            _proprio_batch,
            height_scan_batch,
            privileged_obs_batch,
            proprio_history_batch,
            perception_batch,
            _task_id_batch,
            student_mask_batch,
            _actions_batch,
            _target_values_batch,
            _advantages_batch,
            _returns_batch,
            _old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
        ) in distill_generator:
            if not student_mask_batch.any():
                continue

            mask = student_mask_batch
            with torch.no_grad():
                z_teacher = self.policy.encode_teacher(
                    height_scan_batch[mask],
                    privileged_obs_batch[mask],
                )
            z_student = self.policy.encode_student(
                proprio_history_batch[mask],
                perception_batch[mask],
            )
            distillation_loss = self.policy.distillation_loss(z_student, z_teacher)
            student_loss = self.distillation_loss_coef * distillation_loss

            self.student_optimizer.zero_grad()
            student_loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy.student_parameters()), self.max_grad_norm)
            self.student_optimizer.step()

            mean_distillation_loss += distillation_loss.item()
            num_distill_updates += 1

        loss_dict = {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "distillation": mean_distillation_loss / max(num_distill_updates, 1),
            "router_entropy": mean_router_entropy / num_updates,
            "router_balance": mean_router_balance_loss / num_updates,
            "router_logit_l2": mean_router_logit_l2_loss / num_updates,
            "student_rollout_ratio": self.storage.student_masks.float().mean().item(),
        }
        if self.use_popart:
            loss_dict.update(
                {
                    "value_loss_norm": mean_value_loss / num_updates,
                    "returns_norm_mean": mean_returns_norm_mean / num_updates,
                    "returns_norm_std": mean_returns_norm_std / num_updates,
                }
            )
            stats = self.popart.get_stats()
            for task in range(self.popart.num_tasks):
                loss_dict[f"popart_mean_task_{task}"] = stats["mean"][task].item()
                loss_dict[f"popart_std_task_{task}"] = stats["std"][task].item()

        self.storage.clear()
        return loss_dict

    def _resolve_student_mask(
        self,
        num_envs: int,
        device: torch.device | str,
        student_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

    def _compute_value_loss(
        self,
        value_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        task_id_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.use_popart:
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            return value_loss, {
                "mean": returns_batch.detach().mean(),
                "std": returns_batch.detach().std(unbiased=False),
            }

        returns_norm = self.popart.normalize(task_id_batch, returns_batch)
        old_value_norm = self.popart.normalize(task_id_batch, target_values_batch)
        if self.use_clipped_value_loss:
            value_clipped_norm = old_value_norm + (value_batch - old_value_norm).clamp(
                -self.clip_param,
                self.clip_param,
            )
            value_losses = self._value_loss_elementwise(value_batch, returns_norm)
            value_losses_clipped = self._value_loss_elementwise(value_clipped_norm, returns_norm)
            per_sample_loss = torch.max(value_losses, value_losses_clipped)
        else:
            per_sample_loss = self._value_loss_elementwise(value_batch, returns_norm)

        if self.value_loss_per_task_average:
            value_loss = self._reduce_value_loss_per_task(per_sample_loss, task_id_batch)
        else:
            value_loss = per_sample_loss.mean()

        return value_loss, {
            "mean": returns_norm.detach().mean(),
            "std": returns_norm.detach().std(unbiased=False),
        }

    def _value_loss_elementwise(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.popart_value_loss == "huber":
            return F.huber_loss(
                prediction,
                target,
                reduction="none",
                delta=self.popart_huber_delta,
            )
        return (prediction - target).pow(2)

    def _reduce_value_loss_per_task(self, per_sample_value_loss: torch.Tensor, task_id_batch: torch.Tensor) -> torch.Tensor:
        losses = []
        losses_flat = per_sample_value_loss.view(-1)
        task_ids = task_id_batch.long().view(-1)
        for task in torch.unique(task_ids):
            mask = task_ids == task
            if mask.any():
                losses.append(losses_flat[mask].mean())
        if not losses:
            return losses_flat.mean()
        return torch.stack(losses).mean()

    def _update_learning_rate(
        self,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
        mu_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
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
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _apply_popart_rescale_to_critic_heads(self, old_stats: dict[str, torch.Tensor], new_stats: dict[str, torch.Tensor]):
        value_heads = self.policy.multi_critic.value_heads
        if len(value_heads) != self.popart.num_tasks:
            raise ValueError(f"Expected {self.popart.num_tasks} value heads, got {len(value_heads)}")

        with torch.no_grad():
            for task, head in enumerate(value_heads):
                last_linear = self._get_last_linear(head)
                old_mean = old_stats["mean"][task].to(last_linear.weight.device)
                old_std = old_stats["std"][task].to(last_linear.weight.device)
                new_mean = new_stats["mean"][task].to(last_linear.weight.device)
                new_std = new_stats["std"][task].to(last_linear.weight.device)

                old_std_eff = old_std + self.popart.eps
                new_std_eff = new_std + self.popart.eps
                scale = old_std_eff / new_std_eff
                last_linear.weight.mul_(scale)
                if last_linear.bias is None:
                    raise NotImplementedError("POPArt rescale requires value head last Linear to have bias")
                last_linear.bias.copy_((old_std_eff * last_linear.bias + old_mean - new_mean) / new_std_eff)

    def _get_last_linear(self, module: nn.Module) -> nn.Linear:
        last_linear = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                last_linear = child
        if last_linear is None:
            raise NotImplementedError("POPArt output rescale requires each value head to contain an nn.Linear")
        return last_linear

    def _router_auxiliary_loss(self, router_weights: torch.Tensor, router_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        router_entropy = -(router_weights * torch.log(router_weights + 1e-8)).sum(dim=-1).mean()
        mean_weights = router_weights.mean(dim=0)
        uniform = torch.full_like(mean_weights, 1.0 / mean_weights.numel())
        router_balance_loss = (mean_weights - uniform).pow(2).mean()
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
