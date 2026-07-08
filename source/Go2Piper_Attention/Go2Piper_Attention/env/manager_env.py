from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from . import local_manager


class ManagerRLEnv(ManagerBasedRLEnv):
    """Configuration for the locomotion velocity-tracking environment."""

    TASK_BOX_AVOIDANCE = 0
    TASK_UNDER_TABLE = 1
    TASK_STAIR_UP = 2
    TASK_FLAT = 3
    NUM_TASKS = 4
    TASK_NAMES = (
        "box_avoidance",
        "under_table",
        "stair_up",
        "flat",
    )
    TASK_SCENE_OBJECT_NAMES = (
        "box_obstacle",
        "table_top",
        "table_leg_0",
        "table_leg_1",
        "table_leg_2",
        "table_leg_3",
        "stair_step_0",
        "stair_step_1",
        "stair_step_2",
        "stair_step_3",
        "stair_step_4",
        "stair_step_5",
        "stair_step_6",
        "stair_platform",
    )

    def __init__(self, cfg, render_mode, **kwargs):
        super().__init__(cfg=cfg)
        self._sim_step_counter = 0
        self._cts_moe_enabled = hasattr(self.cfg, "multi_task_rewards")
        self._cts_moe_reward_log: dict[str, torch.Tensor] = {}
        if self._cts_moe_enabled:
            self.robot = self.scene["robot"]
            self._assign_env_tasks()
            self._setup_task_scenes()
            self.prev_base_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self.prev_base_pos[:] = self.robot.data.root_pos_w[:, :3]
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
        self._reset_task_scene(env_ids)
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

    def _is_box_avoidance_enabled(self) -> bool:
        return bool(getattr(self.cfg.multi_task_rewards, "enable_box_avoidance", True))

    def _enabled_task_ids(self) -> list[int]:
        if self._is_box_avoidance_enabled():
            return list(range(self.NUM_TASKS))
        return [self.TASK_UNDER_TABLE, self.TASK_STAIR_UP, self.TASK_FLAT]

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
        if not self._is_box_avoidance_enabled():
            weights[self.TASK_BOX_AVOIDANCE] = 0.0
        if torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("task_sampling_weights must be non-negative and have positive sum")
        probabilities = weights / weights.sum()
        sampled_task_ids = torch.multinomial(probabilities, env_ids.numel(), replacement=True)
        self.task_id[env_ids] = sampled_task_ids.long()

    def _validate_task_id(self, task_id: int):
        if task_id < 0 or task_id >= self.NUM_TASKS:
            raise ValueError(f"fixed_task_id must be in [0, {self.NUM_TASKS - 1}], got {task_id}")
        if task_id == self.TASK_BOX_AVOIDANCE and not self._is_box_avoidance_enabled():
            raise ValueError("box_avoidance is disabled via multi_task_rewards.enable_box_avoidance=False")

    def _refresh_task_masks(self):
        self.mask_box = self.task_id == self.TASK_BOX_AVOIDANCE
        self.mask_under_table = self.task_id == self.TASK_UNDER_TABLE
        self.mask_stair_up = self.task_id == self.TASK_STAIR_UP
        self.mask_flat = self.task_id == self.TASK_FLAT

    def _setup_task_scenes(self):
        self._refresh_task_masks()
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_task_scene(all_env_ids)

    def _reset_task_scene(self, env_ids: torch.Tensor):
        task_ids = self.task_id[env_ids]
        box_env_ids = env_ids[task_ids == self.TASK_BOX_AVOIDANCE]
        table_env_ids = env_ids[task_ids == self.TASK_UNDER_TABLE]
        stair_up_env_ids = env_ids[task_ids == self.TASK_STAIR_UP]
        flat_env_ids = env_ids[task_ids == self.TASK_FLAT]
        self._hide_task_scene_prims(env_ids)
        if self._is_box_avoidance_enabled():
            self._randomize_box_scene(box_env_ids)
        self._randomize_under_table_scene(table_env_ids)
        self._randomize_stair_up_scene(stair_up_env_ids)
        # Flat envs deliberately keep all task-scene objects hidden.
        _ = flat_env_ids
        self._sync_task_scene_transforms()

    def _sync_task_scene_transforms(self):
        if hasattr(self, "scene"):
            self.scene.write_data_to_sim()
        if hasattr(self, "sim"):
            self.sim.forward()

    def _hidden_obstacle_positions(self, count: int) -> torch.Tensor:
        positions = torch.zeros(count, 3, device=self.device)
        positions[:, 2] = -10.0
        return positions

    def _set_obstacle_poses(self, object_name: str, env_ids: torch.Tensor, env_local_positions: torch.Tensor):
        if env_ids.numel() == 0:
            return
        root_state = torch.zeros(env_ids.numel(), 13, device=self.device)
        root_state[:, :3] = env_local_positions + self.scene.env_origins[env_ids]
        root_state[:, 3] = 1.0
        self.scene[object_name].write_root_state_to_sim(root_state, env_ids=env_ids)

    def _hide_task_scene_prims(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        hidden_positions = self._hidden_obstacle_positions(env_ids.numel())
        for name in self.TASK_SCENE_OBJECT_NAMES:
            self._set_obstacle_poses(name, env_ids, hidden_positions)

    def _randomize_box_scene(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        size_z = 0.65
        pos_x = torch.empty(env_ids.numel(), device=self.device).uniform_(1.35, 1.65)
        pos_y = torch.empty(env_ids.numel(), device=self.device).uniform_(-0.35, 0.35)
        pos_z = torch.full((env_ids.numel(),), size_z / 2.0, device=self.device)
        positions = torch.stack([pos_x, pos_y, pos_z], dim=-1)
        self._set_obstacle_poses("box_obstacle", env_ids, positions)

    def _randomize_under_table_scene(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        table_width = 0.95
        table_height = 0.55
        table_length = 1.2
        table_x = torch.empty(env_ids.numel(), device=self.device).uniform_(1.85, 2.15)
        table_y = torch.empty(env_ids.numel(), device=self.device).uniform_(-0.10, 0.10)
        self._set_obstacle_poses(
            "table_top",
            env_ids,
            torch.stack(
                [table_x, table_y, torch.full((env_ids.numel(),), table_height, device=self.device)],
                dim=-1,
            ),
        )
        x_offsets = (-table_length / 2.0 + 0.08, table_length / 2.0 - 0.08)
        y_offsets = (-table_width / 2.0 + 0.08, table_width / 2.0 - 0.08)
        leg_idx = 0
        for x_offset in x_offsets:
            for y_offset in y_offsets:
                self._set_obstacle_poses(
                    f"table_leg_{leg_idx}",
                    env_ids,
                    torch.stack(
                        [
                            table_x + x_offset,
                            table_y + y_offset,
                            torch.full((env_ids.numel(),), table_height / 2.0, device=self.device),
                        ],
                        dim=-1,
                    ),
                )
                leg_idx += 1

    def _randomize_stair_up_scene(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        step_height = 0.15
        step_depth = 0.35
        start_x = torch.empty(env_ids.numel(), device=self.device).uniform_(0.85, 1.05)
        num_steps = 7
        platform_length = 5.0
        for step_idx in range(num_steps):
            height = step_height * (step_idx + 1)
            self._set_obstacle_poses(
                f"stair_step_{step_idx}",
                env_ids,
                torch.stack(
                    [
                        start_x + step_depth * step_idx,
                        torch.zeros(env_ids.numel(), device=self.device),
                        torch.full((env_ids.numel(),), height / 2.0, device=self.device),
                    ],
                    dim=-1,
                ),
            )
        platform_height = step_height * num_steps
        platform_start_x = start_x + (num_steps - 0.5) * step_depth
        self._set_obstacle_poses(
            "stair_platform",
            env_ids,
            torch.stack(
                [
                    platform_start_x + platform_length / 2.0,
                    torch.zeros(env_ids.numel(), device=self.device),
                    torch.full((env_ids.numel(),), platform_height / 2.0, device=self.device),
                ],
                dim=-1,
            ),
        )

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, device=self.device)
        self._task_reward_groups, self._task_reward_logs = self.reward_manager.compute_grouped_by_task_marker(
            dt=self.step_dt
        )

        common_reward, common_logs = self._reward_common()
        box_reward, box_logs = self._reward_box_avoidance()
        table_reward, table_logs = self._reward_under_table()
        stair_up_reward, stair_up_logs = self._reward_stair_up()
        flat_reward, flat_logs = self._reward_flat()

        reward += common_reward

        mask_box = self.task_id == self.TASK_BOX_AVOIDANCE
        mask_table = self.task_id == self.TASK_UNDER_TABLE
        mask_up = self.task_id == self.TASK_STAIR_UP
        mask_flat = self.task_id == self.TASK_FLAT

        reward[mask_box] += box_reward[mask_box]
        reward[mask_table] += table_reward[mask_table]
        reward[mask_up] += stair_up_reward[mask_up]
        reward[mask_flat] += flat_reward[mask_flat]

        self._log_reward_terms(
            common_logs=common_logs,
            box_logs=box_logs,
            table_logs=table_logs,
            stair_up_logs=stair_up_logs,
            flat_logs=flat_logs,
            masks={
                "box": mask_box,
                "under_table": mask_table,
                "stair_up": mask_up,
                "flat": mask_flat,
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

    def _reward_box_avoidance(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = self._task_reward_groups["box_avoidance"].clone()
        logs = {
            f"box/{name}": value
            for name, value in self._task_reward_logs["box_avoidance"].items()
        }
        placeholder = torch.zeros(self.num_envs, device=self.device)
        logs["box/placeholder"] = placeholder
        # TODO: forward progress, obstacle clearance, box collision, center recovery, stuck penalty.
        return reward, logs

    def _reward_under_table(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = self._task_reward_groups["under_table"].clone()
        logs = {
            f"under_table/{name}": value
            for name, value in self._task_reward_logs["under_table"].items()
        }
        placeholder = torch.zeros(self.num_envs, device=self.device)
        logs["under_table/placeholder"] = placeholder
        # TODO: low-body posture, overhead clearance, table collision, posture recovery, arm regularization.
        return reward, logs

    def _reward_stair_up(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward = self._task_reward_groups["stair_up"].clone()
        logs = {
            f"stair_up/{name}": value
            for name, value in self._task_reward_logs["stair_up"].items()
        }
        placeholder = torch.zeros(self.num_envs, device=self.device)
        logs["stair_up/placeholder"] = placeholder
        # TODO: x/z progress, stair height tracking, foot clearance/placement, stability, collision.
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

    def _masked_mean(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return value[mask].mean()
        return torch.zeros((), device=self.device)

    def _log_reward_terms(
        self,
        common_logs: dict[str, torch.Tensor],
        box_logs: dict[str, torch.Tensor],
        table_logs: dict[str, torch.Tensor],
        stair_up_logs: dict[str, torch.Tensor],
        flat_logs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ):
        log = {}
        for name, value in common_logs.items():
            log[f"rew/{name}"] = value.mean()
        for name, value in box_logs.items():
            log[f"rew/{name}"] = self._masked_mean(value, masks["box"])
        for name, value in table_logs.items():
            log[f"rew/{name}"] = self._masked_mean(value, masks["under_table"])
        for name, value in stair_up_logs.items():
            log[f"rew/{name}"] = self._masked_mean(value, masks["stair_up"])
        for name, value in flat_logs.items():
            log[f"rew/{name}"] = self._masked_mean(value, masks["flat"])

        log["task/num_box"] = masks["box"].float().sum()
        log["task/num_under_table"] = masks["under_table"].float().sum()
        log["task/num_stair_up"] = masks["stair_up"].float().sum()
        log["task/num_flat"] = masks["flat"].float().sum()
        self._cts_moe_reward_log = log
        self.extras.setdefault("log", {}).update(log)

    def _attach_task_id_to_obs(self):
        if isinstance(self.obs_buf, dict):
            self.obs_buf["task_id"] = self.task_id

    def _publish_task_extras(self):
        self.extras["task_id"] = self.task_id
        self.extras["task_names"] = self.TASK_NAMES
        self.extras.setdefault("log", {}).update(self._cts_moe_reward_log)

    def _update_reward_buffers(self):
        self.prev_base_pos[:] = self.robot.data.root_pos_w[:, :3]
