# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, quat_apply_inverse, quat_mul, quat_error_magnitude, quat_conjugate, axis_angle_from_quat
from Go2Piper_Attention.tasks.manager_based.go2piper_attention.config.agents.rsl_rl_ppo_cfg import Go2PiperRslRlOnPolicyRunnerCfg, Go2PiperFlatPPORunnerCfg

if TYPE_CHECKING:
    from Go2Piper_Attention.env.manager_env import ManagerBasedRLEnv
    from cfg.command_cfg import UniformPoseCommandCfg


class UniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        
        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w_z = torch.zeros(self.num_envs, 1, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.curr_ee_goal = torch.zeros(self.num_envs, 3, device=self.device)
        self.env = env
        self.count = 0.0
        self.level = 0.0
        if self.cfg.is_Go2ARM_Flat == True:
            cfg_runner = Go2PiperFlatPPORunnerCfg()
            self.num_env_step = cfg_runner.num_steps_per_env
        else:
            # TODO : Rough
            cfg_runner = Go2PiperFlatPPORunnerCfg()
            self.num_env_step = cfg_runner.num_steps_per_env



    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        # return torch.cat([self.pose_command_b[:, :2], self.pose_command_w_z], dim = -1)
        # return self.pose_command_b
        return self.pose_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame

        # print("self.pose_command_b",self.pose_command_b[:, :3])
        # print("self.pose_command_w",self.pose_command_w[:, :3])
        # print("self.pose_command_w_z",self.pose_command_w_z)



        # self.pose_command_w[:, 2] = self.pose_command[:, 2]

        # print("self.pose_command_w",self.pose_command_w[:, :3])
        # if self.cfg.is_Go2ARM or self.cfg.is_Go2ARM_Play == True:
        #     self.pose_command_w[:, 2] = self.pose_command_w_z[:, 0] 
        # # print("pose_command_w",self.pose_command_w)
        # # compute the error    
        # pos_error, rot_error = compute_pose_error(
        #     self.pose_command_w[:, :3],
        #     self.pose_command_w[:, 3:],
        #     self.robot.data.body_state_w[:, self.body_idx, :3],
        #     self.robot.data.body_state_w[:, self.body_idx, 3:7],
        # )
        # print("pos_error_old",pos_error.shape)


        end_effector_curr_pos_b = self.robot.data.body_pos_w[:, self.body_idx] - \
                                    self.robot.data.root_pos_w
        end_effector_curr_pos_b = quat_apply_inverse(self.robot.data.root_state_w[:, 3:7], 
                                                                        end_effector_curr_pos_b)  
        ee_pos_xy_err = torch.abs( end_effector_curr_pos_b[:, :2] - self.pose_command[:, :2] )
        ee_pos_z_err =  torch.abs(self.pose_command[:, 2:3] - self.robot.data.body_pos_w[:, self.body_idx][:, 2:3])
        pos_error = torch.cat([ee_pos_xy_err, ee_pos_z_err], dim=-1)

        des_quat_w = quat_mul(self.robot.data.root_state_w[:, 3:7], self.pose_command[:, 3:7])
        curr_quat_w = self.robot.data.body_state_w[:, self.body_idx, 3:7]  # type: ignore
        source_quat_norm = quat_mul(des_quat_w, quat_conjugate(des_quat_w))[:, 0]
        source_quat_inv = quat_conjugate(des_quat_w) / source_quat_norm.unsqueeze(-1)
        quat_error = quat_mul(curr_quat_w, source_quat_inv) 
        rot_error = axis_angle_from_quat(quat_error)


        # rot_error = quat_error_magnitude(curr_quat_w, des_quat_w)
        
        # print("_update_metrics")
        # print("pose_command_w",self.pose_command_w[:, :3])
        # print("self.robot.data.body_state_w",self.robot.data.body_state_w[:, self.body_idx, :3])
        # print("pos_error",pos_error)
        
        # print("--command--")
        # print("self.pose_command",self.pose_command[:, :3])
        # print("self.robot.data.body_pos_w[:, self.body_idx][:, 2]",self.robot.data.body_pos_w[:, self.body_idx][:, 2].shape)
        # print("self.pose_command",self.pose_command)
        # print("end_effector_curr_pos_b[:, :2]",end_effector_curr_pos_b[:, :2])
        # print("self.robot.data.body_pos_w[:, self.body_idx][:, 2:3]",self.robot.data.body_pos_w[:, self.body_idx][:, 2:3])
        # print("ee_pos_xy_err",ee_pos_xy_err)
        # print("ee_pos_z_err",ee_pos_z_err)   
        # print("pos_error", pos_error)
        # print("pos_error_dis", torch.norm(pos_error, dim=-1) )
        # print("rot_error",rot_error)
        # print("----")

        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        # self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        # self.metrics["level"] = torch.ones(self.num_envs, 1, device=self.device) * self.level
        
        # if self.metrics["position_error"].mean() < 0.08 :
        #     self.count += 1.0
        #     if  self.count / self.num_env_step > 500 and self.level <= self.cfg.max_level:
        #         self.level += 1.0
        #         self.count = 0.0
        # else:
        #      self.count = 0.0


    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # print("_resample_command")
        # -- position



        euler_angles = torch.zeros_like(self.pose_command[env_ids, :3])        
        r = torch.empty(len(env_ids), device=self.device)
        r_1 = torch.empty(1, device=self.device)
        if self.cfg.is_Go2ARM == True:
            ###
            # v1.0
            count = torch.tensor(self.env.common_step_counter / self.num_env_step / self.cfg.curriculum_coeff)
            
            self.pose_command[env_ids, 0] = (r.uniform_(*self.cfg.ranges_init.pos_x))  * torch.clamp((1 - count), 0, 1) + \
                                              (r.uniform_(*self.cfg.ranges_final.pos_x)) * torch.clamp((count), 0, 1)
            self.pose_command[env_ids, 1] = (r.uniform_(*self.cfg.ranges_init.pos_y))  * torch.clamp((1 - count), 0, 1) + \
                                              (r.uniform_(*self.cfg.ranges_final.pos_y)) * torch.clamp((count), 0, 1)
            self.pose_command[env_ids, 2] = (r.uniform_(*self.cfg.ranges_init.pos_z))  * torch.clamp((1 - count), 0, 1) + \
                                              (r.uniform_(*self.cfg.ranges_final.pos_z)) * torch.clamp((count), 0, 1)
            
            # print("self.pose_command",self.pose_command)
            # print("pose_command_shape",self.pose_command.shape)
            # print("self.pose_command[0, 0]",self.pose_command[0,0])
            # self.pose_command_b[env_ids, 2] = self.pose_command_w_z[env_ids, 0]  - torch.clamp( self.robot.data.root_pos_w[env_ids, 2] ,max=0.3)
            # print("self.pose_command_b[env_ids, 2]",self.pose_command_b[env_ids, 2])
            # print("self.pose_command_w_z[env_ids, 0]",self.pose_command_w_z[env_ids, 0])
            # print("( self.robot.data.root_pos_w[env_ids, 2] ) ",( self.robot.data.root_pos_w[env_ids, 2] ) )
            # print("env_ids",env_ids)

            for _, i in enumerate(env_ids):
            
                # print("i",i)
                # print("self.pose_command[i, 0]shape",self.pose_command[i, 0].shape)
                # print("self.pose_command[i, 0]",self.pose_command[i, 0])

                # print("self.pose_command[i, 1]",self.pose_command[i, 1].shape)
                # print("z_dis",z_dis.shape)

                # TODO: length_arm Z-axis need to fix 
                # base limit ( x = 0.35, y = 0.2  z = 0.35 )
                length_arm = torch.norm(torch.stack([self.pose_command[i, 0], 
                                                     self.pose_command[i, 1], 
                                                       ( self.pose_command[i, 2] - torch.clamp(self.robot.data.root_pos_w[i, 2] ,max=0.3) )
                                                    ])) 
                # max_arm_length = 0.73
                while((length_arm > 0.7) or 
                    #   (length_arm < 0.3) or 
                    #   (self.pose_command[i, 0] < 0.4 and torch.abs(self.pose_command[i, 1]) < 0.2 and self.pose_command[i, 0] < 0.3 )
                        (self.pose_command[i, 0] < 0.42 and torch.abs(self.pose_command[i, 1]) < 0.2 and self.pose_command[i, 2] < 0.42)
                     ):
                        self.pose_command[i, 0] = (r_1.uniform_(*self.cfg.ranges_init.pos_x))  * torch.clamp((1 - count), 0, 1) + \
                                                        (r_1.uniform_(*self.cfg.ranges_final.pos_x)) * torch.clamp((count), 0, 1) 
                        self.pose_command[i, 1] = (r_1.uniform_(*self.cfg.ranges_init.pos_y))  * torch.clamp((1 - count), 0, 1) + \
                                                        (r_1.uniform_(*self.cfg.ranges_final.pos_y)) * torch.clamp((count), 0, 1)
                        self.pose_command[i, 2] = (r_1.uniform_(*self.cfg.ranges_init.pos_z))  * torch.clamp((1 - count), 0, 1) + \
                                                         (r_1.uniform_(*self.cfg.ranges_final.pos_z)) * torch.clamp((count), 0, 1)  
           
                        length_arm = torch.norm(torch.stack([self.pose_command[i, 0], 
                                                        self.pose_command[i, 1], 
                                                         ( self.pose_command[i, 2] - torch.clamp(self.robot.data.root_pos_w[i, 2] ,max=0.3) )
                                                        ])) 
            

            delta_x = self.pose_command[env_ids, 0] 
            delta_y = self.pose_command[env_ids, 1] 
            delta_z = self.pose_command[env_ids, 2]  - ( self.robot.data.root_pos_w[env_ids, 2] + 0.1)

            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges_init.roll) * torch.clamp((1 - count), 0, 1) + \
                                 r.uniform_(*self.cfg.ranges_final.roll) * torch.clamp((count), 0, 1)
                                 
            euler_angles[:, 1] = - torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2)) + \
                                    r.uniform_(*self.cfg.ranges.pitch) * torch.clamp((1 - count), 0, 1) + \
                                    r.uniform_(*self.cfg.ranges_final.pitch) * torch.clamp((count), 0, 1) 
                                    
            euler_angles[:, 2] = torch.atan2(delta_y, delta_x) + \
                                r.uniform_(*self.cfg.ranges_init.yaw) * torch.clamp((1 - count), 0, 1) + \
                                r.uniform_(*self.cfg.ranges_final.yaw) *torch.clamp((count), 0, 1)
            
            euler_angles[:, 1] = torch.where(
                (self.pose_command[env_ids, 0] < 0.55) & (self.pose_command[env_ids, 2] < 0.45),
                0.7 * euler_angles[:, 1] + 0.3 * torch.tensor(3.1415926 / 2, device=euler_angles.device, dtype=euler_angles.dtype),
                euler_angles[:, 1] 
            )              

        elif self.cfg.is_Go2ARM_Play == True:
            self.pose_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            self.pose_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1] = r.uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2] = r.uniform_(*self.cfg.ranges.yaw)

            delta_x = self.pose_command[env_ids, 0] 
            delta_y = self.pose_command[env_ids, 1] 
            delta_z = self.pose_command[env_ids, 2]  - ( self.robot.data.root_pos_w[env_ids, 2] + 0.1)

            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges.roll)
                                 
            euler_angles[:, 1] = - torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2)) +  r.uniform_(*self.cfg.ranges.pitch)
                                    
            euler_angles[:, 2] = torch.atan2(delta_y, delta_x) + r.uniform_(*self.cfg.ranges.yaw)

            
            # euler_angles[:, 1] = torch.where(
            #     (self.pose_command[env_ids, 0] < 0.55) & (self.pose_command[env_ids, 2] < 0.4),
            #     0.5 * euler_angles[:, 1] + 0.5 * torch.tensor(3.1415926 / 2, device=euler_angles.device, dtype=euler_angles.dtype),
            #     0.5 * euler_angles[:, 1] + 0.5 * torch.tensor(0.0, device=euler_angles.device, dtype=euler_angles.dtype),
            # )         

            # print(euler_angles[:, 1])

        else:
            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)    
            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1] = r.uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2] = r.uniform_(*self.cfg.ranges.yaw)

        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat     


    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            torch.cat([self.pose_command[:, :2], (self.pose_command[:, 2:3] - torch.clamp(self.robot.data.root_pos_w[:, 2:3] ,max=0.3))],dim = -1),
            self.pose_command[:, 3:],
        )

        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])

