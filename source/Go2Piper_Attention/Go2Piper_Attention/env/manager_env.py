from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse

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

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
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
        self._update_velocity_tracking_curriculum_stats()
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

    def _update_velocity_tracking_curriculum_stats(self):
        curriculum_cfg = getattr(self.cfg, "curriculum", None)
        terrain_term = getattr(curriculum_cfg, "terrain_levels", None) if curriculum_cfg is not None else None
        if terrain_term is None:
            return

        params = getattr(terrain_term, "params", {}) or {}
        command_name = params.get("command_name", "base_velocity")
        min_command_speed = float(params.get("min_command_speed", 0.1))
        asset_cfg = params.get("asset_cfg", None)
        asset_name = getattr(asset_cfg, "name", "robot")

        if not hasattr(self.command_manager, "get_command"):
            return
        try:
            command = self.command_manager.get_command(command_name)
        except (KeyError, AttributeError):
            return
        if command.shape[-1] < 2:
            return

        try:
            asset = self.scene[asset_name]
        except KeyError:
            return
        command_speed = torch.norm(command[:, :2], dim=1)
        vel_error = torch.norm(command[:, :2] - asset.data.root_lin_vel_b[:, :2], dim=1)
        moving_command = command_speed > min_command_speed
        relative_error = vel_error / torch.clamp(command_speed, min=1.0e-6)

        if (
            not hasattr(self, "_terrain_velocity_tracking_error_sum")
            or self._terrain_velocity_tracking_error_sum.shape[0] != self.num_envs
        ):
            self._terrain_velocity_tracking_error_sum = torch.zeros(self.num_envs, device=self.device)
            self._terrain_velocity_tracking_error_count = torch.zeros(self.num_envs, device=self.device)

        self._terrain_velocity_tracking_error_sum += torch.where(
            moving_command, relative_error, torch.zeros_like(relative_error)
        )
        self._terrain_velocity_tracking_error_count += moving_command.float()

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
        task_cfg = self.cfg.multi_task_rewards
        if task_cfg.task_sampling_weights is None:
            return list(range(self.NUM_TASKS))
        weights = torch.as_tensor(task_cfg.task_sampling_weights, dtype=torch.float)
        if weights.numel() != self.NUM_TASKS:
            raise ValueError(f"task_sampling_weights must have length {self.NUM_TASKS}")
        enabled_tasks = torch.nonzero(weights > 0.0, as_tuple=False).view(-1).tolist()
        if len(enabled_tasks) == 0:
            raise ValueError("task_sampling_weights must enable at least one task")
        return [int(task) for task in enabled_tasks]

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

    def _compute_base_height_above_terrain(self, mode: str = "max") -> torch.Tensor:
        """Return base height above local terrain for every env."""
        base_height_w = self.robot.data.root_pos_w[:, 2]
        if hasattr(self.scene, "sensors") and "height_scanner" in self.scene.sensors:
            local_terrain_height_w = self._compute_local_terrain_height_w(mode=mode)
            return base_height_w - local_terrain_height_w
        return base_height_w

    def _compute_local_terrain_height_w(self, mode: str = "mean") -> torch.Tensor:
        """Return local terrain height from the base-mounted height scanner."""
        sensor = self.scene.sensors["height_scanner"]
        terrain_heights_w = sensor.data.ray_hits_w[..., 2]
        valid_hits = torch.isfinite(terrain_heights_w)
        if mode == "mean":
            safe_heights = torch.where(valid_hits, terrain_heights_w, torch.zeros_like(terrain_heights_w))
            valid_counts = valid_hits.sum(dim=1).clamp(min=1)
            return safe_heights.sum(dim=1) / valid_counts
        if mode == "max":
            safe_heights = torch.where(valid_hits, terrain_heights_w, torch.full_like(terrain_heights_w, -torch.inf))
            local_terrain_height_w = torch.max(safe_heights, dim=1).values
            return torch.where(
                torch.isfinite(local_terrain_height_w),
                local_terrain_height_w,
                torch.zeros_like(local_terrain_height_w),
            )
        raise ValueError(f"Unsupported terrain height mode: {mode}")

    def _compute_ee_pose_position_metrics(self, ee_term) -> dict[str, torch.Tensor]:
        """Return EE position errors in base frame and a compatibility z metric."""
        command = ee_term.command
        body_idx = ee_term.body_idx
        ee_pos_w = self.robot.data.body_pos_w[:, body_idx]
        ee_pos_b = quat_apply_inverse(self.robot.data.root_state_w[:, 3:7], ee_pos_w - self.robot.data.root_pos_w)

        base_frame_error = torch.abs(ee_pos_b[:, :3] - command[:, :3])

        return {
            "ee_pose/position_error": torch.norm(base_frame_error, dim=-1),
            "ee_pose/position_error_base_frame": torch.norm(base_frame_error, dim=-1),
            "ee_pose/position_error_xy_b": torch.norm(base_frame_error[:, :2], dim=-1),
            "ee_pose/position_error_z_b": base_frame_error[:, 2],
            "ee_pose/position_error_z_base_frame": base_frame_error[:, 2],
            # Backward-compatible alias for old logs.  The command is base-frame,
            # despite the historical "terrain" name.
            "ee_pose/position_error_z_terrain": base_frame_error[:, 2],
        }

    def _get_per_env_command_metrics(self) -> dict[str, torch.Tensor]:
        """Collect instantaneous command-tracking metrics for each env."""
        metrics: dict[str, torch.Tensor] = {}

        if "ee_pose" in self.command_manager._terms:
            ee_term = self.command_manager.get_term("ee_pose")
            metrics.update(self._compute_ee_pose_position_metrics(ee_term))
            metrics["ee_pose/orientation_error"] = ee_term.metrics["orientation_error"]

        if "base_velocity" in self.command_manager._terms:
            vel_term = self.command_manager.get_term("base_velocity")
            vel_command = vel_term.command
            base_lin_vel = self.robot.data.root_lin_vel_b[:, :2]
            vel_error_xy = vel_command[:, :2] - base_lin_vel
            vel_error_x = torch.abs(vel_error_xy[:, 0])
            vel_error_y = torch.abs(vel_error_xy[:, 1])
            metrics["base_velocity/error_vel_xy"] = torch.norm(
                vel_error_xy,
                dim=-1,
            )
            metrics["base_velocity/error_vel_x"] = vel_error_x
            metrics["base_velocity/error_vel_y"] = vel_error_y
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
        }

        log: dict[str, torch.Tensor] = {}
        for task_name, mask in task_masks.items():
            for metric_name, values in per_env_metrics.items():
                log[f"Metrics/{task_name}/{metric_name}"] = self._masked_mean(values, mask)
            log.update(self._get_base_velocity_reward_metrics(task_name, mask))
        log.update(self._get_floating_ring_height_metrics(task_masks["floating_ring"]))
        log.update(self._get_terrain_level_metrics())
        log.update(self._get_base_velocity_curriculum_metrics())

        self._cts_moe_task_metrics_log = log

    def _get_base_velocity_reward_metrics(self, task_name: str, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return reward-shaped diagnostics for the base-velocity tracker."""
        if "base_velocity" not in self.command_manager._terms:
            return {}

        reward_cfg = getattr(self.cfg, "rewards", None)
        if reward_cfg is None:
            return {}

        x_term = getattr(reward_cfg, f"track_lin_vel_x_exp_{task_name}", None)
        y_term = getattr(reward_cfg, f"track_lin_vel_y_exp_{task_name}", None)
        if x_term is None or y_term is None:
            return {}

        vel_term = self.command_manager.get_term("base_velocity")
        vel_command = vel_term.command
        lin_vel = self.robot.data.root_lin_vel_b[:, :2]

        error_x = torch.abs(vel_command[:, 0] - lin_vel[:, 0])
        error_y = torch.abs(vel_command[:, 1] - lin_vel[:, 1])

        x_params = getattr(x_term, "params", {}) or {}
        y_params = getattr(y_term, "params", {}) or {}
        x_std = max(float(x_params.get("std", 0.2)), 1.0e-6)
        y_std = max(float(y_params.get("std", 0.2)), 1.0e-6)
        x_weight = float(getattr(x_term, "weight", 1.0))
        y_weight = float(getattr(y_term, "weight", 1.0))

        x_raw = torch.exp(-error_x / x_std)
        y_raw = torch.exp(-error_y / y_std)

        return {
            f"Metrics/{task_name}/base_velocity/reward_track_lin_vel_x_exp_raw": self._masked_mean(x_raw, mask),
            f"Metrics/{task_name}/base_velocity/reward_track_lin_vel_y_exp_raw": self._masked_mean(y_raw, mask),
            f"Metrics/{task_name}/base_velocity/reward_track_lin_vel_x_exp_weighted": self._masked_mean(
                x_raw * x_weight, mask
            ),
            f"Metrics/{task_name}/base_velocity/reward_track_lin_vel_y_exp_weighted": self._masked_mean(
                y_raw * y_weight, mask
            ),
        }

    def _get_floating_ring_height_metrics(self, floating_ring_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return diagnostics matching the floating-ring base-height reward regions."""
        terrain = getattr(self.scene, "terrain", None)
        if terrain is None or not hasattr(terrain, "terrain_types") or not hasattr(terrain, "terrain_levels"):
            return {}

        reward_term = getattr(self.cfg.rewards, "track_base_height_exp_floating_ring", None)
        params = getattr(reward_term, "params", {}) or {}
        inside_target = float(params.get("desired_height", 0.22))
        outside_target = float(params.get("outside_desired_height", 0.28))
        terrain_height_mode = params.get("terrain_height_mode", "max")
        floating_ring_terrain_type = int(params.get("floating_ring_terrain_type", self.TASK_FLOATING_RING))
        platform_width = float(params.get("platform_width", 2.0))
        ring_width_range = params.get("ring_width_range", (0.6, 1.8))
        difficulty_range = params.get("difficulty_range", (0.0, 1.0))
        margin = float(params.get("margin", 0.0))

        terrain_levels = terrain.terrain_levels.float()
        max_level = max(float(getattr(terrain, "max_terrain_level", 1) - 1), 1.0)
        lower, upper = difficulty_range
        difficulty = lower + (upper - lower) * torch.clamp(terrain_levels / max_level, 0.0, 1.0)

        ring_width = ring_width_range[0] + difficulty * (ring_width_range[1] - ring_width_range[0])
        inner_half_width = 0.5 * platform_width - margin
        outer_half_width = 0.5 * platform_width + ring_width + margin

        local_xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        abs_xy = torch.abs(local_xy)
        inside_outer = (abs_xy[:, 0] <= outer_half_width) & (abs_xy[:, 1] <= outer_half_width)
        outside_inner = (abs_xy[:, 0] >= inner_half_width) | (abs_xy[:, 1] >= inner_half_width)
        on_floating_ring = terrain.terrain_types == floating_ring_terrain_type
        inside_ring = on_floating_ring & inside_outer & outside_inner
        outside_ring = on_floating_ring & ~inside_ring

        height_above_terrain = self._compute_base_height_above_terrain(mode=terrain_height_mode)
        target_height = torch.where(
            inside_ring,
            torch.full_like(height_above_terrain, inside_target),
            torch.full_like(height_above_terrain, outside_target),
        )
        height_error = torch.abs(target_height - height_above_terrain)

        def scalar(value: float) -> torch.Tensor:
            return torch.tensor(value, dtype=torch.float, device=self.device)

        return {
            "Metrics/floating_ring/base_height/inside_target": scalar(inside_target),
            "Metrics/floating_ring/base_height/outside_target": scalar(outside_target),
            "Metrics/floating_ring/base_height/inside_ratio": self._masked_mean(inside_ring.float(), floating_ring_mask),
            "Metrics/floating_ring/base_height/height_above_terrain_inside": self._masked_mean(
                height_above_terrain, inside_ring
            ),
            "Metrics/floating_ring/base_height/height_above_terrain_outside": self._masked_mean(
                height_above_terrain, outside_ring
            ),
            "Metrics/floating_ring/base_height/error_inside": self._masked_mean(height_error, inside_ring),
            "Metrics/floating_ring/base_height/error_outside": self._masked_mean(height_error, outside_ring),
            "Metrics/floating_ring/base_height/error_mixed": self._masked_mean(height_error, floating_ring_mask),
        }

    def _get_base_velocity_curriculum_metrics(self) -> dict[str, torch.Tensor]:
        """Return command curriculum scalars for logging."""
        if "base_velocity" not in self.command_manager._terms:
            return {}

        vel_term = self.command_manager.get_term("base_velocity")
        cfg = getattr(vel_term, "cfg", None)
        if cfg is None:
            return {}

        def scalar(value: float | int | bool) -> torch.Tensor:
            return torch.tensor(float(value), dtype=torch.float, device=self.device)

        command = vel_term.command
        log: dict[str, torch.Tensor] = {
            "Curriculum/base_velocity/command_x_mean": command[:, 0].mean(),
            "Curriculum/base_velocity/command_y_mean": command[:, 1].mean(),
            "Curriculum/base_velocity/command_yaw_mean": command[:, 2].mean(),
            "Curriculum/base_velocity/command_xy_norm_mean": torch.norm(command[:, :2], dim=-1).mean(),
            "Curriculum/base_velocity/standing_ratio": getattr(vel_term, "is_standing_env", torch.zeros_like(command[:, 0])).float().mean(),
        }

        enabled = bool(getattr(cfg, "is_Go2ARM", False))
        log["Curriculum/base_velocity/enabled"] = scalar(enabled)

        curriculum_coeff = getattr(cfg, "curriculum_coeff", None)
        num_env_step = max(float(getattr(vel_term, "num_env_step", 1)), 1.0)
        if enabled and curriculum_coeff is not None:
            raw_progress = float(self.common_step_counter) / (num_env_step * max(float(curriculum_coeff), 1.0))
            progress = min(max(raw_progress, 0.0), 1.0)
            log["Curriculum/base_velocity/progress_raw"] = scalar(raw_progress)
            log["Curriculum/base_velocity/progress"] = scalar(progress)
            log["Curriculum/base_velocity/curriculum_coeff"] = scalar(float(curriculum_coeff))
        else:
            progress = 0.0
            log["Curriculum/base_velocity/progress_raw"] = scalar(0.0)
            log["Curriculum/base_velocity/progress"] = scalar(0.0)

        ranges_init = getattr(cfg, "ranges_init", None)
        ranges_final = getattr(cfg, "ranges_final", None)
        ranges = getattr(cfg, "ranges", None)
        axes = (
            ("lin_vel_x", "x"),
            ("lin_vel_y", "y"),
            ("ang_vel_z", "yaw"),
        )

        for attr_name, log_name in axes:
            if enabled and ranges_init is not None and ranges_final is not None:
                init_min, init_max = getattr(ranges_init, attr_name)
                final_min, final_max = getattr(ranges_final, attr_name)
                current_min = float(init_min) * (1.0 - progress) + float(final_min) * progress
                current_max = float(init_max) * (1.0 - progress) + float(final_max) * progress
            elif ranges is not None:
                current_min, current_max = getattr(ranges, attr_name)
            else:
                continue
            log[f"Curriculum/base_velocity/current_{log_name}_min"] = scalar(current_min)
            log[f"Curriculum/base_velocity/current_{log_name}_max"] = scalar(current_max)

        return log

    def _get_terrain_level_metrics(self) -> dict[str, torch.Tensor]:
        """Return per-terrain curriculum level metrics for logging."""
        terrain = getattr(self.scene, "terrain", None)
        if terrain is None or not hasattr(terrain, "terrain_levels") or not hasattr(terrain, "terrain_types"):
            return {}

        terrain_levels = terrain.terrain_levels.float()
        terrain_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        terrain_types = (
            self._terrain_column_task_ids(terrain_cfg)[terrain.terrain_types.long()]
            if terrain_cfg is not None
            else terrain.terrain_types.long()
        )
        terrain_names = self._context_task_names()

        log: dict[str, torch.Tensor] = {}
        log["Curriculum/terrain_level/mean"] = terrain_levels.mean()
        for terrain_type, terrain_name in enumerate(terrain_names):
            mask = terrain_types == terrain_type
            log[f"Curriculum/terrain_level/{terrain_name}"] = self._masked_mean(terrain_levels, mask)
        return log

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
            tile_offset = torch.floor((progress + 0.5 * tile_size) / tile_size).long()
            if terrain is not None and hasattr(terrain, "terrain_types"):
                base_task_id = self._terrain_column_task_ids(terrain_cfg)[terrain.terrain_types.long()].to(
                    device=progress.device
                )
            else:
                base_task_id = torch.zeros_like(tile_offset)
            task_id = base_task_id + tile_offset
            return torch.clamp(task_id, 0, self.NUM_TASKS - 1)
        if (
            terrain is not None
            and hasattr(terrain, "terrain_types")
            and terrain_cfg is not None
        ):
            return self._terrain_column_task_ids(terrain_cfg)[terrain.terrain_types.long()]
        if hasattr(self, "task_id"):
            return self.task_id
        return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _terrain_column_task_ids(self, terrain_cfg) -> torch.Tensor:
        proportions = torch.tensor(
            [float(sub_cfg.proportion) for sub_cfg in terrain_cfg.sub_terrains.values()],
            dtype=torch.float,
            device=self.device,
        )
        if proportions.numel() == 0 or torch.any(proportions < 0.0) or proportions.sum() <= 0.0:
            return torch.arange(int(terrain_cfg.num_cols), dtype=torch.long, device=self.device).clamp(
                max=self.NUM_TASKS - 1
            )
        cumulative = torch.cumsum(proportions / proportions.sum(), dim=0)
        column_task_ids = []
        for column in range(int(terrain_cfg.num_cols)):
            threshold = float(column) / float(terrain_cfg.num_cols) + 0.001
            task_id = torch.nonzero(threshold < cumulative, as_tuple=False)[0, 0]
            column_task_ids.append(task_id)
        return torch.stack(column_task_ids).long().clamp(max=self.NUM_TASKS - 1)

    def _context_task_names(self) -> tuple[str, ...]:
        terrain = getattr(self.scene, "terrain", None)
        terrain_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        if terrain_cfg is not None and terrain_cfg.num_cols == len(terrain_cfg.sub_terrains):
            return tuple(terrain_cfg.sub_terrains.keys())
        return self.TASK_NAMES

    def _update_reward_buffers(self):
        self.prev_base_pos[:] = self.robot.data.root_pos_w[:, :3]
