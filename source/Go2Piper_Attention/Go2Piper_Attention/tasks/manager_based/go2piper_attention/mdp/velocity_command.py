
"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log # type:ignore

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from Go2Piper_Attention.tasks.manager_based.go2piper_attention.config.agents.rsl_rl_ppo_cfg import Go2PiperFlatPPORunnerCfg

if TYPE_CHECKING:
    from Go2Piper_Attention.env.manager_env import ManagerBasedRLEnv

    from cfg.command_cfg import UniformVelocityCommandCfg


class UniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # self.env = env ###
        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.curr_base_goal = torch.zeros(self.num_envs, 3, device=self.device)
        self.env = env
        if self.cfg.is_Go2ARM_Flat == True:
            cfg_runner = Go2PiperFlatPPORunnerCfg()
            self.num_env_step = cfg_runner.num_steps_per_env
        else:
            cfg_runner = Go2PiperFlatPPORunnerCfg()
            self.num_env_step = cfg_runner.num_steps_per_env



    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self.count = torch.tensor(
            self.env.common_step_counter / self.num_env_step / self.cfg.curriculum_coeff,
            device=self.device,
        )
    
        
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):


        r = torch.empty(len(env_ids), device=self.device)
        if self.cfg.is_Go2ARM:
            self.count = torch.tensor(
                self.env.common_step_counter / self.num_env_step / self.cfg.curriculum_coeff,
                device=self.device,
            )

            self._sample_curriculum_velocity(env_ids, self.cfg.ranges_init, self.cfg.ranges_final, self.count)
            self._resample_flat_task_command(env_ids, self.count)
 
        else:
            self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
            self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
            self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)  

        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _sample_curriculum_velocity(self, env_ids: Sequence[int], ranges_init, ranges_final, count: torch.Tensor):
        r = torch.empty(len(env_ids), device=self.device)
        init_weight = torch.clamp(1 - count, 0, 1)
        final_weight = torch.clamp(count, 0, 1)
        self.vel_command_b[env_ids, 0] = (
            r.uniform_(*ranges_init.lin_vel_x) * init_weight
            + r.uniform_(*ranges_final.lin_vel_x) * final_weight
        )
        self.vel_command_b[env_ids, 1] = (
            r.uniform_(*ranges_init.lin_vel_y) * init_weight
            + r.uniform_(*ranges_final.lin_vel_y) * final_weight
        )
        self.vel_command_b[env_ids, 2] = (
            r.uniform_(*ranges_init.ang_vel_z) * init_weight
            + r.uniform_(*ranges_final.ang_vel_z) * final_weight
        )

    def _resample_flat_task_command(self, env_ids: Sequence[int], count: torch.Tensor):
        if self.cfg.flat_ranges_init is None or self.cfg.flat_ranges_final is None:
            return
        if not hasattr(self.env, "task_id") or not hasattr(self.env, "TASK_FLAT"):
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        flat_env_ids = env_ids_tensor[self.env.task_id[env_ids_tensor] == self.env.TASK_FLAT]
        if flat_env_ids.numel() == 0:
            return
        self._sample_curriculum_velocity(flat_env_ids, self.cfg.flat_ranges_init, self.cfg.flat_ranges_final, count)
        
    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """

        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)

            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _init_adaptive_curriculum(self):
        """初始化自适应课程学习器"""
        class SmoothAdaptiveCurriculum:
            def __init__(self, initial_value=0.0, max_value=1.0, smoothing_factor=0.9, 
                        error_smoothing=0.95, improvement_threshold=0.02, base_step=0.01):
                self.current_value = initial_value
                self.max_value = max_value
                self.smoothing_factor = smoothing_factor
                self.error_smoothing = error_smoothing
                self.improvement_threshold = improvement_threshold
                self.base_step = base_step
                
                self.smoothed_error = None
                self.smoothed_velocity = 0.0
                self.previous_error = None
                
            def update(self, current_error):
                # 误差平滑
                if self.smoothed_error is None:
                    self.smoothed_error = current_error
                    # print(f"=curr{self.smoothed_error:.3f}")
                else:
                    self.smoothed_error = (self.error_smoothing * self.smoothed_error + 
                                        (1 - self.error_smoothing) * current_error)
                    # print(f"smoothed_error: {self.smoothed_error:.3f}")

                # 计算目标速度
                error_change = 0.0
                if self.previous_error is not None:
                    error_change = self.previous_error - current_error  # 正=改进

                if error_change > self.improvement_threshold:
                    # 明显改进：正向加速
                    acceleration = min(error_change / self.previous_error, 2.0)
                    target_velocity = self.base_step * (1.0 + acceleration)
                    
                elif error_change < -self.improvement_threshold:
                    # 明显退步：减速
                    target_velocity = -self.base_step * 0.5
                else:
                    # 稳定状态
                    target_velocity = self.base_step
                
                # 平滑更新
                self.smoothed_velocity = (self.smoothing_factor * self.smoothed_velocity + 
                                        (1 - self.smoothing_factor) * target_velocity)
                
                # 更新课程参数
                self.current_value = torch.clamp(
                    torch.tensor(self.current_value + self.smoothed_velocity), 
                    0.0, self.max_value
                ).item()
                self.previous_error = current_error
                
                return self.current_value
        
        # 初始化课程学习器
        self.adaptive_curriculum = SmoothAdaptiveCurriculum(
            initial_value=0.0,
            max_value=1.0,
            smoothing_factor=0.9,      # 保守平滑
            error_smoothing=0.95,      # 误差平滑
            improvement_threshold=0.05, # 10%改进阈值
            base_step = 0.001,        # 基础步长
        )

    def _get_current_velocity_error(self):
        """获取当前速度跟踪误差"""
        # 计算当前速度命令与实际速度的误差
        lin_vel_error = torch.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], 
            dim=-1
        )
        ang_vel_error = torch.abs(
            self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        )
        
        # 组合误差（可以加权平均）
        total_error = 0.7 * lin_vel_error.mean() + 0.3 * ang_vel_error.mean()
        
        return total_error.item()