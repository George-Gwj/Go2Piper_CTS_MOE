from __future__ import annotations

from collections.abc import Sequence

import torch

from .manager_env import ManagerRLEnv


class ManagerRLEnvSeparate(ManagerRLEnv):
    """Manager env variant that exposes decoupled leg and arm rewards.

    This keeps the original ``ManagerRLEnv`` untouched.  For CTS-MoE multitask
    configs, reward terms are first selected by their task suffix marker
    (``_common``, ``_flat``, ...), then split by name prefix:

    - terms starting with ``end_effector`` contribute to arm rewards;
    - all other marked terms contribute to leg rewards.
    """

    ARM_REWARD_PREFIX = "end_effector"

    def step(self, action):
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        if self._cts_moe_enabled:
            self.leg_reward_buf, self.arm_reward_buf = self._get_separate_rewards()
            self.reward_buf = self.leg_reward_buf + self.arm_reward_buf
            self._publish_separate_reward_extras()
        else:
            self.reward_buf, self.arm_reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        if self._cts_moe_enabled:
            self._attach_task_id_to_obs()
            self._publish_task_extras()
            self._publish_separate_reward_extras()
            self._update_reward_buffers()

        return self.obs_buf, self.reward_buf, self.arm_reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def reset(self, seed=None, env_ids: Sequence[int] | None = None, options: dict | None = None):
        obs, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        if self._cts_moe_enabled and hasattr(self, "leg_reward_buf") and hasattr(self, "arm_reward_buf"):
            self._publish_separate_reward_extras()
            extras = self.extras
        return obs, extras

    def _get_separate_rewards(self) -> tuple[torch.Tensor, torch.Tensor]:
        branch_groups, branch_logs = self._compute_grouped_rewards_by_task_and_branch(dt=self.step_dt)

        leg_common, arm_common, common_logs = self._reward_branch_common(branch_groups, branch_logs)
        leg_flat, arm_flat, flat_logs = self._reward_branch_task("flat", branch_groups, branch_logs)
        leg_ascend, arm_ascend, ascend_logs = self._reward_branch_task("ascend", branch_groups, branch_logs)
        leg_descend, arm_descend, descend_logs = self._reward_branch_task("descend", branch_groups, branch_logs)
        leg_floating, arm_floating, floating_logs = self._reward_branch_task("floating_ring", branch_groups, branch_logs)
        leg_rough, arm_rough, rough_logs = self._reward_branch_task("rough", branch_groups, branch_logs)

        leg_reward = leg_common.clone()
        arm_reward = arm_common.clone()

        task_id = self._context_task_id()
        masks = {
            "flat": task_id == self.TASK_FLAT,
            "ascend": task_id == self.TASK_ASCEND,
            "descend": task_id == self.TASK_DESCEND,
            "floating_ring": task_id == self.TASK_FLOATING_RING,
            "rough": task_id == self.TASK_ROUGH,
        }

        leg_by_task = {
            "flat": leg_flat,
            "ascend": leg_ascend,
            "descend": leg_descend,
            "floating_ring": leg_floating,
            "rough": leg_rough,
        }
        arm_by_task = {
            "flat": arm_flat,
            "ascend": arm_ascend,
            "descend": arm_descend,
            "floating_ring": arm_floating,
            "rough": arm_rough,
        }
        for task_name, mask in masks.items():
            leg_reward[mask] += leg_by_task[task_name][mask]
            arm_reward[mask] += arm_by_task[task_name][mask]

        self._log_separate_reward_terms(
            common_logs=common_logs,
            flat_logs=flat_logs,
            ascend_logs=ascend_logs,
            descend_logs=descend_logs,
            floating_ring_logs=floating_logs,
            rough_logs=rough_logs,
            masks=masks,
            leg_reward=leg_reward,
            arm_reward=arm_reward,
        )
        return leg_reward, arm_reward

    def _compute_grouped_rewards_by_task_and_branch(
        self,
        dt: float,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, dict[str, torch.Tensor]]]]:
        branch_groups = {
            group: {
                "leg": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "arm": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            }
            for group, _suffix in self.reward_manager.TASK_REWARD_SUFFIXES
        }
        branch_logs = {group: {"leg": {}, "arm": {}} for group, _suffix in self.reward_manager.TASK_REWARD_SUFFIXES}

        for term_idx, (name, term_cfg) in enumerate(
            zip(self.reward_manager._term_names, self.reward_manager._term_cfgs)
        ):
            group_name, clean_name = self.reward_manager._classify_reward_term(name)
            if group_name is None:
                self.reward_manager._step_reward[:, term_idx] = 0.0
                continue
            if term_cfg.weight == 0.0:
                value = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                self.reward_manager._step_reward[:, term_idx] = 0.0
            else:
                value = term_cfg.func(self, **term_cfg.params) * term_cfg.weight * dt
                value = self.reward_manager._mask_value_for_reward_group(group_name, value)
                self.reward_manager._step_reward[:, term_idx] = value / dt
                self.reward_manager._episode_sums[name] += value

            branch = "arm" if clean_name.startswith(self.ARM_REWARD_PREFIX) else "leg"
            branch_groups[group_name][branch] += value
            branch_logs[group_name][branch][clean_name] = value

        return branch_groups, branch_logs

    def _reward_branch_common(
        self,
        branch_groups: dict[str, dict[str, torch.Tensor]],
        branch_logs: dict[str, dict[str, dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        leg_reward = branch_groups["common"]["leg"].clone()
        arm_reward = branch_groups["common"]["arm"].clone()
        logs = self._format_branch_logs("common", branch_logs)

        r_alive = torch.ones(self.num_envs, device=self.device) * self.cfg.multi_task_rewards.alive_weight
        leg_reward += r_alive
        logs["common/leg/alive"] = r_alive
        logs["common/leg/marked_total"] = branch_groups["common"]["leg"]
        logs["common/arm/marked_total"] = branch_groups["common"]["arm"]
        return leg_reward, arm_reward, logs

    def _reward_branch_task(
        self,
        task_name: str,
        branch_groups: dict[str, dict[str, torch.Tensor]],
        branch_logs: dict[str, dict[str, dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        logs = self._format_branch_logs(task_name, branch_logs)
        logs[f"{task_name}/leg/marked_total"] = branch_groups[task_name]["leg"]
        logs[f"{task_name}/arm/marked_total"] = branch_groups[task_name]["arm"]
        return branch_groups[task_name]["leg"].clone(), branch_groups[task_name]["arm"].clone(), logs

    def _format_branch_logs(
        self,
        task_name: str,
        branch_logs: dict[str, dict[str, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        logs = {}
        for branch_name, values in branch_logs[task_name].items():
            for reward_name, value in values.items():
                logs[f"{task_name}/{branch_name}/{reward_name}"] = value
        return logs

    def _log_separate_reward_terms(
        self,
        common_logs: dict[str, torch.Tensor],
        flat_logs: dict[str, torch.Tensor],
        ascend_logs: dict[str, torch.Tensor],
        descend_logs: dict[str, torch.Tensor],
        floating_ring_logs: dict[str, torch.Tensor],
        rough_logs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        leg_reward: torch.Tensor,
        arm_reward: torch.Tensor,
    ):
        log = {}
        for name, value in common_logs.items():
            log[f"rew/{name}"] = value.mean()

        task_logs = {
            "flat": flat_logs,
            "ascend": ascend_logs,
            "descend": descend_logs,
            "floating_ring": floating_ring_logs,
            "rough": rough_logs,
        }
        for task_name, logs in task_logs.items():
            for name, value in logs.items():
                log[f"rew/{name}"] = self._masked_mean(value, masks[task_name])

        log["rew/leg_total"] = leg_reward.mean()
        log["rew/arm_total"] = arm_reward.mean()
        log["rew/total"] = (leg_reward + arm_reward).mean()
        log["task/num_flat"] = masks["flat"].float().sum()
        log["task/num_ascend"] = masks["ascend"].float().sum()
        log["task/num_descend"] = masks["descend"].float().sum()
        log["task/num_floating_ring"] = masks["floating_ring"].float().sum()
        log["task/num_rough"] = masks["rough"].float().sum()
        self._cts_moe_reward_log = log
        self.extras.setdefault("log", {}).update(log)

    def _publish_separate_reward_extras(self):
        if hasattr(self, "leg_reward_buf"):
            self.extras["leg_rewards"] = self.leg_reward_buf
            self.extras["reward_leg"] = self.leg_reward_buf
        if hasattr(self, "arm_reward_buf"):
            self.extras["arm_rewards"] = self.arm_reward_buf
            self.extras["reward_arm"] = self.arm_reward_buf
