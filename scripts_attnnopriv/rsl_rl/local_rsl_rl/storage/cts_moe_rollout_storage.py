from __future__ import annotations

import torch


class CTSMoERolloutStorage:
    """Rollout storage for StructureAwareCTSMoEPolicy.

    The storage keeps structured observations instead of the legacy flat
    actor/critic observation pairs.  It also supports per-task advantage
    normalization for sparse multi-critic training.
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
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
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
        actions_shape: tuple[int, ...] | list[int],
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

        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
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
        self.rewards[self.step].copy_(self._finite(transition.rewards).view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1).bool())
        self.values[self.step].copy_(self._finite(transition.values))
        self.actions_log_prob[self.step].copy_(self._finite(transition.actions_log_prob).view(-1, 1))
        self.mu[self.step].copy_(self._finite(transition.action_mean))
        self.sigma[self.step].copy_(self._finite(transition.action_sigma))
        self.router_weights[self.step].copy_(self._finite(transition.router_weights))
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
        per_task_advantage_normalization: bool = True,
    ):
        last_values = self._finite(last_values)
        self.rewards = self._finite(self.rewards)
        self.values = self._finite(self.values)
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.returns = self._finite(self.returns)
        self.advantages = self._finite(self.advantages)
        if normalize_advantage:
            if per_task_advantage_normalization:
                self._normalize_advantages_per_task()
            else:
                self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std(unbiased=False) + 1e-8)

    def _normalize_advantages_per_task(self):
        flat_adv = self.advantages.view(-1)
        flat_task = self.task_ids.view(-1)
        normalized = flat_adv.clone()
        for task in torch.unique(flat_task):
            mask = flat_task == task
            task_adv = flat_adv[mask]
            normalized[mask] = (task_adv - task_adv.mean()) / (task_adv.std(unbiased=False) + 1e-8)
        self.advantages.copy_(normalized.view_as(self.advantages))

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
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

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
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
