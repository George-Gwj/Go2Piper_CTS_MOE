# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

from Go2Piper_Attention.env.manager_env import ManagerRLEnv
from isaaclab.envs import DirectRLEnv
from rsl_rl.env import VecEnv


class RslRlVecEnvWrapper(VecEnv):
    """CTS-MoE-only IsaacLab wrapper for RSL-RL.

    The wrapper exposes structured observations for `StructureAwareCTSMoEPolicy`:
    `proprio`, `proprio_history`, `privileged_obs`, `height_scan`, `perception`,
    and `task_id`.
    """

    def __init__(self, env: ManagerRLEnv | DirectRLEnv, clip_actions: float | None = None):
        if not isinstance(env.unwrapped, ManagerRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must inherit from ManagerRLEnv or DirectRLEnv. "
                f"Environment type: {type(env)}"
            )

        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        if not hasattr(self.unwrapped, "observation_manager"):
            raise ValueError("CTS-MoE wrapper requires an IsaacLab observation_manager.")
        group_dims = self.unwrapped.observation_manager.group_obs_dim
        required_groups = ("proprio", "proprio_history", "privileged_obs", "height_scan", "depth")
        missing_groups = [name for name in required_groups if name not in group_dims]
        if missing_groups:
            raise ValueError(f"CTS-MoE observation groups are missing: {missing_groups}")

        self.proprio_dim = group_dims["proprio"][0]
        self.privileged_obs_dim = group_dims["privileged_obs"][0]
        self.proprio_history_flat_dim = group_dims["proprio_history"][0]
        configured_history_length = getattr(self.unwrapped.cfg.observations.proprio_history, "history_length", None)
        self.proprio_history_length = int(configured_history_length or 5)
        self.proprio_history_dim = self.proprio_history_flat_dim // self.proprio_history_length

        self._modify_action_space()
        self.env.reset()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerRLEnv | DirectRLEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_observations(self) -> dict[str, torch.Tensor]:
        return self.get_cts_moe_observations()

    def get_cts_moe_observations(self) -> dict[str, torch.Tensor]:
        obs_dict = self.unwrapped.observation_manager.compute()
        return self._extract_cts_moe_observations(obs_dict)

    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        obs_dict, extras = self.env.reset()
        structured_obs = self._extract_cts_moe_observations(obs_dict, extras=extras)
        extras["observations"] = obs_dict
        return structured_obs, extras

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        return self.step_cts_moe(actions)

    def step_cts_moe(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, reward, _unused_reward, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        structured_obs = self._extract_cts_moe_observations(obs_dict, extras=extras)
        return structured_obs, reward, dones, extras

    def close(self):
        return self.env.close()

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions,
            high=self.clip_actions,
            shape=(self.num_actions,),
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space,
            self.num_envs,
        )

    def _extract_cts_moe_observations(self, obs_dict: dict, extras: dict | None = None) -> dict[str, torch.Tensor]:
        proprio = obs_dict["proprio"]
        proprio_history = self._reshape_proprio_history(obs_dict["proprio_history"])
        privileged_obs = obs_dict["privileged_obs"]
        height_scan = self._extract_single_term_group(obs_dict["height_scan"], "height_scan")
        perception = self._extract_single_term_group(obs_dict["depth"], "depth_image")

        task_id = obs_dict.get("task_id")
        if task_id is None and extras is not None:
            task_id = extras.get("task_id")
        if task_id is None:
            task_id = self.unwrapped.task_id

        return {
            "proprio": proprio,
            "proprio_history": proprio_history,
            "privileged_obs": privileged_obs,
            "height_scan": height_scan,
            "perception": perception,
            "task_id": task_id.long(),
        }

    def _extract_single_term_group(self, group_obs, term_name: str) -> torch.Tensor:
        if isinstance(group_obs, dict):
            return group_obs[term_name]
        return group_obs

    def _reshape_proprio_history(self, proprio_history: torch.Tensor) -> torch.Tensor:
        if proprio_history.dim() == 3:
            return proprio_history
        if proprio_history.dim() != 2:
            raise ValueError(f"proprio_history must be [B, H*D] or [B, H, D], got {tuple(proprio_history.shape)}")

        term_names, term_lengths = self._get_obs_list_length("proprio_history")
        if not term_lengths:
            return proprio_history.view(self.num_envs, self.proprio_history_length, self.proprio_history_dim)

        history_chunks = []
        start = 0
        for length in term_lengths:
            end = start + length
            term_flat = proprio_history[:, start:end]
            term_dim = length // self.proprio_history_length
            history_chunks.append(term_flat.view(self.num_envs, self.proprio_history_length, term_dim))
            start = end
        return torch.cat(history_chunks, dim=-1)

    def _get_obs_list_length(self, obs_group: str) -> tuple[list[str], list[int]]:
        active_terms = self.unwrapped.observation_manager.get_active_iterable_terms(0)
        keys = [item[0] for item in active_terms]
        lengths = [len(item[1]) for item in active_terms]
        output_keys = [key for key in keys if key.startswith(obs_group)]
        output_lens = [length for key, length in zip(keys, lengths) if key.startswith(obs_group)]
        return output_keys, [int(length) for length in output_lens]
