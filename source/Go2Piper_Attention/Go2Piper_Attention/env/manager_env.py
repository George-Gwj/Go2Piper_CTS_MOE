from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from . import local_manager


class ManagerRLEnv(ManagerBasedRLEnv):
    """Configuration for the locomotion velocity-tracking environment."""

    TASK_FLAT = 0
    TASK_ASCEND = 1
    TASK_DESCEND = 2
    TASK_FLOATING_RING = 3
    TASK_ROUGH = 4
    NUM_TASKS = 5
    TASK_NAMES = (
        "flat",
        "ascend",
        "descend",
        "floating_ring",
        "rough",
    )

    def __init__(self, cfg, render_mode, **kwargs):
        super().__init__(cfg=cfg)
        self._sim_step_counter = 0
        self._cts_moe_enabled = hasattr(self.cfg, "multi_task_rewards")
        self._cts_moe_reward_log: dict[str, torch.Tensor] = {}
        if self._cts_moe_enabled:
            self.robot = self.scene["robot"]
            self._assign_env_tasks()
            self.prev_base_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self.prev_base_pos[:] = self.robot.data.root_pos_w[:, :3]
            self._cts_moe_task_metrics_log: dict[str, torch.Tensor] = {}
            self._publish_task_extras()


    def load_managers(self):
        super().load_managers()
        self.reward_manager = local_manager.RewardManager(self.cfg.rewards, self)
        self.observation_manager = local_manager.ObservationManager(self.cfg.observations,self)

    #TODO:
    def step(self, action) :
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        if self._cts_moe_enabled:
            self.reward_buf = self._get_rewards()
            self.arm_reward_buf = torch.zeros_like(self.reward_buf)
        else:
            self.reward_buf, self.arm_reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)

        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history = True)
        if self._cts_moe_enabled:
            self._attach_task_id_to_obs()
            self._publish_task_extras()
            self._update_reward_buffers()

        return self.obs_buf, self.reward_buf, self.arm_reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def reset(self, seed=None, env_ids: Sequence[int] | None = None, options: dict | None = None):
        obs, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        if self._cts_moe_enabled:
            self.obs_buf = obs
            self._attach_task_id_to_obs()
            self._publish_task_extras()
            obs = self.obs_buf
            extras = self.extras
        return obs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if not self._cts_moe_enabled:
            return

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        task_cfg = self.cfg.multi_task_rewards
        if not task_cfg.fixed_task_assignment:
            self._sample_task_ids(env_ids)
            self._refresh_task_masks()
        self.prev_base_pos[env_ids] = self.robot.data.root_pos_w[env_ids, :3]

    def _assign_env_tasks(self):
        task_cfg = self.cfg.multi_task_rewards
        if task_cfg.fixed_task_id is not None:
            fixed_task_id = int(task_cfg.fixed_task_id)
            self._validate_task_id(fixed_task_id)
            self.task_id = torch.full((self.num_envs,), fixed_task_id, dtype=torch.long, device=self.device)
            self._refresh_task_masks()
            return

        if not task_cfg.fixed_task_assignment:
            self.task_id = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
            self._sample_task_ids(torch.arange(self.num_envs, dtype=torch.long, device=self.device))
            self._refresh_task_masks()
            return

        enabled_tasks = self._enabled_task_ids()
        num_enabled = len(enabled_tasks)
        task_id = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        start = 0
        base_count = self.num_envs // num_enabled
        remainder = self.num_envs % num_enabled
        for idx, task in enumerate(enabled_tasks):
            count = base_count + (1 if idx < remainder else 0)
            task_id[start : start + count] = task
            start += count
        self.task_id = task_id
        self._refresh_task_masks()

    def _enabled_task_ids(self) -> list[int]:
        return list(range(self.NUM_TASKS))

    def _sample_task_ids(self, env_ids: torch.Tensor):
        task_cfg = self.cfg.multi_task_rewards
        if task_cfg.fixed_task_id is not None:
            fixed_task_id = int(task_cfg.fixed_task_id)
            self._validate_task_id(fixed_task_id)
            self.task_id[env_ids] = fixed_task_id
            return

        if task_cfg.task_sampling_weights is None:
            enabled_tasks = self._enabled_task_ids()
            sampled_indices = torch.randint(
                low=0,
                high=len(enabled_tasks),
                size=(env_ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )
            enabled_task_ids = torch.tensor(enabled_tasks, dtype=torch.long, device=self.device)
            self.task_id[env_ids] = enabled_task_ids[sampled_indices]
            return

        weights = torch.tensor(task_cfg.task_sampling_weights, dtype=torch.float, device=self.device)
        if weights.numel() != self.NUM_TASKS:
            raise ValueError(f"task_sampling_weights must have length {self.NUM_TASKS}")
        if torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("task_sampling_weights must be non-negative and have positive sum")
        probabilities = weights / weights.sum()
        sampled_task_ids = torch.multinomial(probabilities, env_ids.numel(), replacement=True)
        self.task_id[env_ids] = sampled_task_ids.long()

    def _validate_task_id(self, task_id: int):
        if task_id < 0 or task_id >= self.NUM_TASKS:
            raise ValueError(f"fixed_task_id must be in [0, {self.NUM_TASKS - 1}], got {task_id}")

    def _refresh_task_masks(self):
        task_id = self._context_task_id()
        self.mask_flat = task_id == self.TASK_FLAT
        self.mask_ascend = task_id == self.TASK_ASCEND
        self.mask_descend = task_id == self.TASK_DESCEND
        self.mask_floating_ring = task_id == self.TASK_FLOATING_RING
        self.mask_rough = task_id == self.TASK_ROUGH

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, device=self.device)
        self._task_reward_groups, self._task_reward_logs = self.reward_manager.compute_grouped_by_task_marker(
            dt=self.step_dt
        )

        common_reward, common_logs = self._reward_common()
        flat_reward, flat_logs = self._reward_flat()
        ascend_reward, ascend_logs = self._reward_ascend()
        descend_reward, descend_logs = self._reward_descend()
        floating_ring_reward, floating_ring_logs = self._reward_floating_ring()
        rough_reward, rough_logs = self._reward_rough()

        reward += common_reward

        task_id = self._context_task_id()
        mask_flat = task_id == self.TASK_FLAT
        mask_ascend = task_id == self.TASK_ASCEND
        mask_descend = task_id == self.TASK_DESCEND
        mask_floating_ring = task_id == self.TASK_FLOATING_RING
        mask_rough = task_id == self.TASK_ROUGH

        reward[mask_flat] += flat_reward[mask_flat]
        reward[mask_ascend] += ascend_reward[mask_ascend]
        reward[mask_descend] += descend_reward[mask_descend]
        reward[mask_floating_ring] += floating_ring_reward[mask_floating_ring]
        reward[mask_rough] += rough_reward[mask_rough]

        self._log_reward_terms(
            common_logs=common_logs,
            flat_logs=flat_logs,
            ascend_logs=ascend_logs,
            descend_logs=descend_logs,
            floating_ring_logs=floating_ring_logs,
            rough_logs=rough_logs,
            masks={
                "flat": mask_flat,
                "ascend": mask_ascend,
                "descend": mask_descend,
                "floating_ring": mask_floating_ring,
                "rough": mask_rough,
            },
        )
        return reward

    def _reward_common(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = self._task_reward_groups["common"].clone()
        logs = {
            f"common/{name}": value
            for name, value in self._task_reward_logs["common"].items()
        }
        logs["common/marked_total"] = reward

        r_alive = torch.ones(self.num_envs, device=self.device) * self.cfg.multi_task_rewards.alive_weight
        reward += r_alive
        logs["common/alive"] = r_alive
        return reward, logs

    def _reward_flat(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = self._task_reward_groups["flat"].clone()
        logs = {
            f"flat/{name}": value
            for name, value in self._task_reward_logs["flat"].items()
        }
        placeholder = torch.zeros(self.num_envs, device=self.device)
        logs["flat/placeholder"] = placeholder
        # TODO: flat-terrain progress, velocity tracking, stability, obstacle-free locomotion.
        return reward, logs

    def _reward_ascend(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = self._task_reward_groups["ascend"].clone()
        logs = {
            f"ascend/{name}": value
            for name, value in self._task_reward_logs["ascend"].items()
        }
        logs["ascend/placeholder"] = torch.zeros(self.num_envs, device=self.device)
        return reward, logs

    def _reward_descend(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = self._task_reward_groups["descend"].clone()
        logs = {
            f"descend/{name}": value
            for name, value in self._task_reward_logs["descend"].items()
        }
        logs["descend/placeholder"] = torch.zeros(self.num_envs, device=self.device)
        return reward, logs

    def _reward_floating_ring(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = self._task_reward_groups["floating_ring"].clone()
        logs = {
            f"floating_ring/{name}": value
            for name, value in self._task_reward_logs["floating_ring"].items()
        }
        logs["floating_ring/placeholder"] = torch.zeros(self.num_envs, device=self.device)
        return reward, logs

    def _reward_rough(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = self._task_reward_groups["rough"].clone()
        logs = {
            f"rough/{name}": value
            for name, value in self._task_reward_logs["rough"].items()
        }
        logs["rough/placeholder"] = torch.zeros(self.num_envs, device=self.device)
        return reward, logs

    def _masked_mean(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return value[mask].mean()
        return torch.zeros((), device=self.device)

    def _compute_base_height_above_terrain(self) -> torch.Tensor:
        """Return base height above local terrain for every env."""
        base_height_w = self.robot.data.root_pos_w[:, 2]
        if hasattr(self.scene, "sensors") and "height_scanner" in self.scene.sensors:
            sensor = self.scene.sensors["height_scanner"]
            terrain_heights_w = sensor.data.ray_hits_w[..., 2]
            valid_hits = torch.isfinite(terrain_heights_w)
            safe_heights = torch.where(valid_hits, terrain_heights_w, torch.full_like(terrain_heights_w, -torch.inf))
            local_terrain_height_w = torch.max(safe_heights, dim=1).values
            local_terrain_height_w = torch.where(
                torch.isfinite(local_terrain_height_w),
                local_terrain_height_w,
                torch.zeros_like(local_terrain_height_w),
            )
            return base_height_w - local_terrain_height_w
        return base_height_w

    def _get_per_env_command_metrics(self) -> dict[str, torch.Tensor]:
        """Collect instantaneous command-tracking metrics for each env."""
        metrics: dict[str, torch.Tensor] = {}

        if "ee_pose" in self.command_manager._terms:
            ee_term = self.command_manager.get_term("ee_pose")
            metrics["ee_pose/position_error"] = ee_term.metrics["position_error"]
            metrics["ee_pose/orientation_error"] = ee_term.metrics["orientation_error"]

        if "base_velocity" in self.command_manager._terms:
            vel_term = self.command_manager.get_term("base_velocity")
            vel_command = vel_term.command
            metrics["base_velocity/error_vel_xy"] = torch.norm(
                vel_command[:, :2] - self.robot.data.root_lin_vel_b[:, :2],
                dim=-1,
            )
            metrics["base_velocity/error_vel_yaw"] = torch.abs(
                vel_command[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
            )

        metrics["base/height_w"] = self.robot.data.root_pos_w[:, 2]
        metrics["base/height_above_terrain"] = self._compute_base_height_above_terrain()
        return metrics

    def _log_task_metrics(self):
        """Publish per-task command and base-height metrics to extras['log']."""
        per_env_metrics = self._get_per_env_command_metrics()
        task_masks = {
            "flat": self.mask_flat,
            "ascend": self.mask_ascend,
            "descend": self.mask_descend,
            "floating_ring": self.mask_floating_ring,
            "rough": self.mask_rough,
        }

        log: dict[str, torch.Tensor] = {}
        for task_name, mask in task_masks.items():
            for metric_name, values in per_env_metrics.items():
                log[f"Metrics/{task_name}/{metric_name}"] = self._masked_mean(values, mask)

        self._cts_moe_task_metrics_log = log

    def _log_reward_terms(
        self,
        common_logs: dict[str, torch.Tensor],
        flat_logs: dict[str, torch.Tensor],
        ascend_logs: dict[str, torch.Tensor],
        descend_logs: dict[str, torch.Tensor],
        floating_ring_logs: dict[str, torch.Tensor],
        rough_logs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ):
        log = {}
        for name, value in common_logs.items():
            log[f"rew/{name}"] = value.mean()
        for name, value in flat_logs.items():
            log[f"rew/{name}"] = self._masked_mean(value, masks["flat"])
        for name, value in ascend_logs.items():
            log[f"rew/{name}"] = self._masked_mean(value, masks["ascend"])
        for name, value in descend_logs.items():
            log[f"rew/{name}"] = self._masked_mean(value, masks["descend"])
        for name, value in floating_ring_logs.items():
            log[f"rew/{name}"] = self._masked_mean(value, masks["floating_ring"])
        for name, value in rough_logs.items():
            log[f"rew/{name}"] = self._masked_mean(value, masks["rough"])

        log["task/num_flat"] = masks["flat"].float().sum()
        log["task/num_ascend"] = masks["ascend"].float().sum()
        log["task/num_descend"] = masks["descend"].float().sum()
        log["task/num_floating_ring"] = masks["floating_ring"].float().sum()
        log["task/num_rough"] = masks["rough"].float().sum()
        self._cts_moe_reward_log = log
        self.extras.setdefault("log", {}).update(log)

    def _attach_task_id_to_obs(self):
        if isinstance(self.obs_buf, dict):
            self.obs_buf["task_id"] = self._context_task_id()

    def _publish_task_extras(self):
        self.extras["task_id"] = self._context_task_id()
        self.extras["task_names"] = self._context_task_names()
        if self._cts_moe_enabled:
            self._log_task_metrics()
        self.extras.setdefault("log", {}).update(self._cts_moe_reward_log)
        self.extras.setdefault("log", {}).update(self._cts_moe_task_metrics_log)

    def _context_task_id(self) -> torch.Tensor:
        terrain = getattr(self.scene, "terrain", None)
        terrain_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        task_cfg = getattr(self.cfg, "multi_task_rewards", None)
        if (
            task_cfg is not None
            and getattr(task_cfg, "play_long_terrain", False)
            and terrain_cfg is not None
            and hasattr(self, "robot")
        ):
            tile_size = terrain_cfg.size[1] if getattr(task_cfg, "play_long_axis", "y") == "y" else terrain_cfg.size[0]
            local_pos = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
            progress = local_pos[:, 1] if getattr(task_cfg, "play_long_axis", "y") == "y" else local_pos[:, 0]
            task_id = torch.floor((progress + 0.5 * tile_size) / tile_size).long()
            return torch.clamp(task_id, 0, self.NUM_TASKS - 1)
        if (
            terrain is not None
            and hasattr(terrain, "terrain_types")
            and terrain_cfg is not None
            and terrain_cfg.num_cols == len(terrain_cfg.sub_terrains)
        ):
            return terrain.terrain_types
        return self.task_id

    def _context_task_names(self) -> tuple[str, ...]:
        terrain = getattr(self.scene, "terrain", None)
        terrain_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        if terrain_cfg is not None and terrain_cfg.num_cols == len(terrain_cfg.sub_terrains):
            return tuple(terrain_cfg.sub_terrains.keys())
        return self.TASK_NAMES

    def _update_reward_buffers(self):
        self.prev_base_pos[:] = self.robot.data.root_pos_w[:, :3]
