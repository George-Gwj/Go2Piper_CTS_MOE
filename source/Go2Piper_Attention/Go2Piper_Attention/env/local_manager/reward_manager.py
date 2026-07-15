from __future__ import annotations
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

    def compute(self, dt: float) -> tuple[torch.Tensor,torch.Tensor]:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        self.arm_reward_buf[:] = 0.0 
        # self.count = torch.tensor(self.env.common_step_counter / self.num_env_step / self.curriculum_coeff)

        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # check if the term is a special term for arm

            if name.startswith("end_effector"):  ## TODO: 
                self.arm_reward_buf += value
                # self._reward_buf += value * 0.2 * torch.clamp(self.count,min=0.0,max=1.0)
            else:
                self._reward_buf += value
                # self.arm_reward_buf += value * 0.1 * torch.clamp(self.count,min=0.0,max=1.0)

            # self.arm_reward_buf += value
            # self._reward_buf += value

            # update episodic sum
            self._episode_sums[name] += value

            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt

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
