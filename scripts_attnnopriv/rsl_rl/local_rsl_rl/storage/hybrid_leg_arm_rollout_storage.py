from __future__ import annotations

import torch


class HybridLegArmRolloutStorage:
    """Rollout storage for decoupled leg CTS-MoE and arm MLP PPO updates.

    The environment rollout is still collected jointly because leg and arm
    dynamics are coupled, but PPO bookkeeping is separated so leg rewards update
    only the leg branch and arm rewards update only the arm branch.
    """

    class Transition:
        def __init__(self):
            self.proprio = None
            self.height_scan = None
            self.privileged_obs = None
            self.proprio_history = None
            self.perception = None
            self.task_id = None
            self.student_mask = None
            self.actions = None
            self.leg_actions = None
            self.arm_actions = None
            self.leg_rewards = None
            self.arm_rewards = None
            self.dones = None
            self.leg_values = None
            self.arm_values = None
            self.leg_actions_log_prob = None
            self.arm_actions_log_prob = None
            self.leg_action_mean = None
            self.leg_action_sigma = None
            self.arm_action_mean = None
            self.arm_action_sigma = None
            self.router_weights = None

        def clear(self):
            self.__init__()

    @staticmethod
    def _finite(tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(tensor):
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        proprio_shape: tuple[int, ...] | list[int],
        height_scan_shape: tuple[int, ...] | list[int],
        privileged_obs_shape: tuple[int, ...] | list[int],
        proprio_history_shape: tuple[int, ...] | list[int],
        perception_shape: tuple[int, ...] | list[int],
        leg_actions_shape: tuple[int, ...] | list[int],
        arm_actions_shape: tuple[int, ...] | list[int],
        num_experts: int,
        device: str = "cpu",
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0

        self.proprio = torch.zeros(num_transitions_per_env, num_envs, *proprio_shape, device=device)
        self.height_scan = torch.zeros(num_transitions_per_env, num_envs, *height_scan_shape, device=device)
        self.privileged_obs = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=device)
        self.proprio_history = torch.zeros(num_transitions_per_env, num_envs, *proprio_history_shape, device=device)
        self.perception = torch.zeros(num_transitions_per_env, num_envs, *perception_shape, device=device)
        self.task_ids = torch.zeros(num_transitions_per_env, num_envs, dtype=torch.long, device=device)
        self.student_masks = torch.zeros(num_transitions_per_env, num_envs, dtype=torch.bool, device=device)

        self.leg_actions = torch.zeros(num_transitions_per_env, num_envs, *leg_actions_shape, device=device)
        self.arm_actions = torch.zeros(num_transitions_per_env, num_envs, *arm_actions_shape, device=device)
        self.actions = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *(list(leg_actions_shape[:-1]) + [leg_actions_shape[-1] + arm_actions_shape[-1]]),
            device=device,
        )

        self.leg_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.arm_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=device)

        self.leg_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.arm_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.leg_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.arm_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.leg_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.arm_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

        self.leg_actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.arm_actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.leg_mu = torch.zeros(num_transitions_per_env, num_envs, *leg_actions_shape, device=device)
        self.leg_sigma = torch.zeros(num_transitions_per_env, num_envs, *leg_actions_shape, device=device)
        self.arm_mu = torch.zeros(num_transitions_per_env, num_envs, *arm_actions_shape, device=device)
        self.arm_sigma = torch.zeros(num_transitions_per_env, num_envs, *arm_actions_shape, device=device)
        self.router_weights = torch.zeros(num_transitions_per_env, num_envs, num_experts, device=device)

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow. Call clear() before adding more transitions.")

        self.proprio[self.step].copy_(self._finite(transition.proprio))
        self.height_scan[self.step].copy_(self._finite(transition.height_scan))
        self.privileged_obs[self.step].copy_(self._finite(transition.privileged_obs))
        self.proprio_history[self.step].copy_(self._finite(transition.proprio_history))
        self.perception[self.step].copy_(self._finite(transition.perception))
        self.task_ids[self.step].copy_(transition.task_id.long().view(-1))
        self.student_masks[self.step].copy_(transition.student_mask.bool().view(-1))

        self.actions[self.step].copy_(self._finite(transition.actions))
        self.leg_actions[self.step].copy_(self._finite(transition.leg_actions))
        self.arm_actions[self.step].copy_(self._finite(transition.arm_actions))
        self.leg_rewards[self.step].copy_(self._finite(transition.leg_rewards).view(-1, 1))
        self.arm_rewards[self.step].copy_(self._finite(transition.arm_rewards).view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1).bool())
        self.leg_values[self.step].copy_(self._finite(transition.leg_values))
        self.arm_values[self.step].copy_(self._finite(transition.arm_values))
        self.leg_actions_log_prob[self.step].copy_(self._finite(transition.leg_actions_log_prob).view(-1, 1))
        self.arm_actions_log_prob[self.step].copy_(self._finite(transition.arm_actions_log_prob).view(-1, 1))
        self.leg_mu[self.step].copy_(self._finite(transition.leg_action_mean))
        self.leg_sigma[self.step].copy_(self._finite(transition.leg_action_sigma))
        self.arm_mu[self.step].copy_(self._finite(transition.arm_action_mean))
        self.arm_sigma[self.step].copy_(self._finite(transition.arm_action_sigma))
        self.router_weights[self.step].copy_(self._finite(transition.router_weights))
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(
        self,
        last_leg_values: torch.Tensor,
        last_arm_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
        per_task_leg_advantage_normalization: bool = True,
        normalize_arm_advantage: bool = True,
    ):
        self._compute_branch_returns(
            rewards=self.leg_rewards,
            values=self.leg_values,
            returns=self.leg_returns,
            advantages=self.leg_advantages,
            last_values=last_leg_values,
            gamma=gamma,
            lam=lam,
        )
        self._compute_branch_returns(
            rewards=self.arm_rewards,
            values=self.arm_values,
            returns=self.arm_returns,
            advantages=self.arm_advantages,
            last_values=last_arm_values,
            gamma=gamma,
            lam=lam,
        )

        if normalize_advantage:
            if per_task_leg_advantage_normalization:
                self._normalize_advantages_per_task(self.leg_advantages)
            else:
                self._normalize_advantages_global(self.leg_advantages)
            if normalize_arm_advantage:
                self._normalize_advantages_global(self.arm_advantages)

    def _compute_branch_returns(
        self,
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
    ):
        last_values = self._finite(last_values)
        rewards.copy_(self._finite(rewards))
        values.copy_(self._finite(values))
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns[step] = advantage + values[step]

        advantages.copy_(self._finite(returns - values))
        returns.copy_(self._finite(returns))

    def _normalize_advantages_global(self, advantages: torch.Tensor):
        advantages.copy_((advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8))

    def _normalize_advantages_per_task(self, advantages: torch.Tensor):
        flat_adv = advantages.view(-1)
        flat_task = self.task_ids.view(-1)
        normalized = flat_adv.clone()
        for task in torch.unique(flat_task):
            mask = flat_task == task
            task_adv = flat_adv[mask]
            normalized[mask] = (task_adv - task_adv.mean()) / (task_adv.std(unbiased=False) + 1e-8)
        advantages.copy_(normalized.view_as(advantages))

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        proprio = self.proprio.flatten(0, 1)
        height_scan = self.height_scan.flatten(0, 1)
        privileged_obs = self.privileged_obs.flatten(0, 1)
        proprio_history = self.proprio_history.flatten(0, 1)
        perception = self.perception.flatten(0, 1)
        task_ids = self.task_ids.flatten(0, 1)
        student_masks = self.student_masks.flatten(0, 1)

        leg_actions = self.leg_actions.flatten(0, 1)
        arm_actions = self.arm_actions.flatten(0, 1)
        leg_values = self.leg_values.flatten(0, 1)
        arm_values = self.arm_values.flatten(0, 1)
        leg_returns = self.leg_returns.flatten(0, 1)
        arm_returns = self.arm_returns.flatten(0, 1)
        leg_advantages = self.leg_advantages.flatten(0, 1)
        arm_advantages = self.arm_advantages.flatten(0, 1)
        old_leg_actions_log_prob = self.leg_actions_log_prob.flatten(0, 1)
        old_arm_actions_log_prob = self.arm_actions_log_prob.flatten(0, 1)
        old_leg_mu = self.leg_mu.flatten(0, 1)
        old_leg_sigma = self.leg_sigma.flatten(0, 1)
        old_arm_mu = self.arm_mu.flatten(0, 1)
        old_arm_sigma = self.arm_sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    proprio[batch_idx],
                    height_scan[batch_idx],
                    privileged_obs[batch_idx],
                    proprio_history[batch_idx],
                    perception[batch_idx],
                    task_ids[batch_idx],
                    student_masks[batch_idx],
                    leg_actions[batch_idx],
                    arm_actions[batch_idx],
                    leg_values[batch_idx],
                    arm_values[batch_idx],
                    leg_advantages[batch_idx],
                    arm_advantages[batch_idx],
                    leg_returns[batch_idx],
                    arm_returns[batch_idx],
                    old_leg_actions_log_prob[batch_idx],
                    old_arm_actions_log_prob[batch_idx],
                    old_leg_mu[batch_idx],
                    old_leg_sigma[batch_idx],
                    old_arm_mu[batch_idx],
                    old_arm_sigma[batch_idx],
                )
