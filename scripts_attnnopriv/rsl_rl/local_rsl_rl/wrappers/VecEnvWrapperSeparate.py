# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch

from Go2Piper_Attention.env.manager_env_separate import ManagerRLEnvSeparate
from isaaclab.envs import DirectRLEnv

from .VecEnvWrapper import RslRlVecEnvWrapper


class RslRlVecEnvWrapperSeparate(RslRlVecEnvWrapper):
    """CTS-MoE wrapper variant that preserves decoupled leg/arm rewards."""

    def __init__(self, env: ManagerRLEnvSeparate | DirectRLEnv, clip_actions: float | None = None):
        if not isinstance(env.unwrapped, ManagerRLEnvSeparate) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The separate environment must inherit from ManagerRLEnvSeparate or DirectRLEnv. "
                f"Environment type: {type(env)}"
            )
        super().__init__(env, clip_actions=clip_actions)

    @property
    def unwrapped(self) -> ManagerRLEnvSeparate | DirectRLEnv:
        return self.env.unwrapped

    def step_cts_moe(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, reward, arm_reward, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        arm_reward = arm_reward.to(reward.device).view_as(reward)
        leg_reward = extras.get("leg_rewards")
        if leg_reward is None:
            leg_reward = reward - arm_reward
        leg_reward = leg_reward.to(reward.device).view_as(reward)

        extras["leg_rewards"] = leg_reward
        extras["arm_rewards"] = arm_reward
        extras["reward_leg"] = leg_reward
        extras["reward_arm"] = arm_reward

        structured_obs = self._extract_cts_moe_observations(obs_dict, extras=extras)
        return structured_obs, reward, dones, extras

    def _extract_cts_moe_observations(self, obs_dict: dict, extras: dict | None = None) -> dict[str, torch.Tensor]:
        if "leg_proprio" not in obs_dict or "arm_proprio" not in obs_dict:
            return super()._extract_cts_moe_observations(obs_dict, extras=extras)

        leg_proprio = obs_dict["leg_proprio"]
        arm_proprio = obs_dict["arm_proprio"]
        proprio = torch.cat([leg_proprio, arm_proprio], dim=-1)

        if "leg_proprio_history" in obs_dict and "arm_proprio_history" in obs_dict:
            leg_history = self._reshape_history_group(obs_dict["leg_proprio_history"], "leg_proprio_history")
            arm_history = self._reshape_history_group(obs_dict["arm_proprio_history"], "arm_proprio_history")
            proprio_history = torch.cat([leg_history, arm_history], dim=-1)
        else:
            proprio_history = self._reshape_proprio_history(obs_dict["proprio_history"])

        privileged_obs = obs_dict["privileged_obs"]
        height_scan = self._extract_single_term_group(obs_dict["height_scan"], "height_scan")
        perception = self._extract_single_term_group(obs_dict["depth"], "depth_image")

        task_id = obs_dict.get("task_id")
        if task_id is None and extras is not None:
            task_id = extras.get("task_id")
        if task_id is None:
            task_id = self.unwrapped.task_id

        structured_obs = {
            "proprio": proprio,
            "proprio_history": proprio_history,
            "privileged_obs": privileged_obs,
            "height_scan": height_scan,
            "perception": perception,
            "task_id": task_id.long(),
        }
        return structured_obs

    def _reshape_history_group(self, history: torch.Tensor, obs_group: str) -> torch.Tensor:
        if history.dim() == 3:
            return history
        if history.dim() != 2:
            raise ValueError(f"{obs_group} must be [B, H*D] or [B, H, D], got {tuple(history.shape)}")

        term_names, term_lengths = self._get_obs_list_length(obs_group)
        history_length = self.proprio_history_length
        if not term_lengths:
            history_dim = history.shape[-1] // history_length
            return history.view(self.num_envs, history_length, history_dim)

        history_chunks = []
        start = 0
        for length in term_lengths:
            end = start + length
            term_flat = history[:, start:end]
            term_dim = length // history_length
            history_chunks.append(term_flat.view(self.num_envs, history_length, term_dim))
            start = end
        return torch.cat(history_chunks, dim=-1)
