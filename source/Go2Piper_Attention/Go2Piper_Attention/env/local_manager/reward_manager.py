from __future__ import annotations
from collections.abc import Sequence

import torch
from isaaclab.managers import RewardManager as RewardManagerBase
from Go2Piper_Attention.tasks.manager_based.go2piper_attention.config.agents.rsl_rl_ppo_cfg import Go2PiperRslRlOnPolicyRunnerCfg, Go2PiperFlatPPORunnerCfg

class RewardManager(RewardManagerBase):
    TASK_REWARD_SUFFIXES = (
        ("common", "_common"),
        ("flat", "_flat"),
        ("ascend", "_ascend"),
        ("descend", "_descend"),
        ("floating_ring", "_floating_ring"),
        ("rough", "_rough"),
    )

    def __init__(self,cfg, env):
        super().__init__(cfg, env)
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.arm_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) 
        self.curriculum_coeff = 4000

        cfg_runner = Go2PiperFlatPPORunnerCfg()
        self.num_env_step = cfg_runner.num_steps_per_env
        self.env = env

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return episodic reward logs without diluting task-specific terms."""
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for key in self._episode_sums.keys():
            group_name, _clean_name = self._classify_reward_term(key)
            values = self._episode_sums[key][env_ids]

            if group_name is not None and group_name != "common":
                task_mask = self._task_group_mask(group_name)
                if task_mask is not None:
                    task_mask = task_mask[env_ids]
                    if not task_mask.any():
                        self._episode_sums[key][env_ids] = 0.0
                        continue
                    values = values[task_mask]

            episodic_sum_avg = torch.mean(values)
            extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras

    def compute(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rewards by task suffix only.

        Terms ending with ``_common`` are applied to all environments.  Terms
        ending with a task suffix such as ``_flat`` or ``_rough`` are applied
        only to environments whose current task id matches that suffix.
        """
        self._reward_buf[:] = 0.0
        self.arm_reward_buf[:] = 0.0

        grouped_rewards, _grouped_logs = self.compute_grouped_by_task_marker(dt=dt)
        for group_name, _suffix in self.TASK_REWARD_SUFFIXES:
            self._reward_buf += grouped_rewards[group_name]

        return self._reward_buf, self.arm_reward_buf

    def compute_grouped_by_task_marker(
        self,
        dt: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """Compute reward terms and group them by their task suffix marker.

        Reward term names ending with:
        - ``_common`` are used for all tasks.
        - ``_flat`` are used only for flat terrain envs.
        - ``_ascend`` are used only for ascending stair terrain envs.
        - ``_descend`` are used only for descending stair terrain envs.
        - ``_floating_ring`` are used only for floating-ring terrain envs.
        - ``_rough`` are used only for rough terrain envs.
        """
        grouped_rewards = {
            group: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for group, _suffix in self.TASK_REWARD_SUFFIXES
        }
        grouped_logs = {group: {} for group, _suffix in self.TASK_REWARD_SUFFIXES}

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            group_name, clean_name = self._classify_reward_term(name)
            if group_name is None:
                self._step_reward[:, term_idx] = 0.0
                continue
            if term_cfg.weight == 0.0:
                value = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                self._step_reward[:, term_idx] = 0.0
            else:
                value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
                value = self._mask_value_for_reward_group(group_name, value)
                self._step_reward[:, term_idx] = value / dt
                self._episode_sums[name] += value
            grouped_rewards[group_name] += value
            grouped_logs[group_name][clean_name] = value

        return grouped_rewards, grouped_logs

    def _classify_reward_term(self, name: str) -> tuple[str | None, str]:
        for group_name, suffix in self.TASK_REWARD_SUFFIXES:
            if name.endswith(suffix):
                return group_name, name[: -len(suffix)]
        return None, name

    def _mask_value_for_reward_group(self, group_name: str, value: torch.Tensor) -> torch.Tensor:
        mask = self._task_group_mask(group_name)
        if mask is None:
            return value
        return torch.where(mask, value, torch.zeros_like(value))

    def _task_group_mask(self, group_name: str) -> torch.Tensor | None:
        if group_name == "common":
            return None

        task_id = self.env._context_task_id()
        task_constants = {
            "flat": self.env.TASK_FLAT,
            "ascend": self.env.TASK_ASCEND,
            "descend": self.env.TASK_DESCEND,
            "floating_ring": self.env.TASK_FLOATING_RING,
            "rough": self.env.TASK_ROUGH,
        }
        if group_name not in task_constants:
            return None
        return task_id == task_constants[group_name]
