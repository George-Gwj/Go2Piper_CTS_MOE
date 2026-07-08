# Copyright (c) 2022-2024, The Isaac Attention Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from Go2Piper_Attention.tasks.manager_based.go2piper_attention.go2piper_cts_moe_env_cfg import (
    LocomotionVelocityEnvCfg,
)
from Go2Piper_Attention.assets.go2arm_articulation_cfg import GO2PIPER_CFG
from Go2Piper_Attention.tasks.manager_based.go2piper_attention.mdp import command_cfg


@configclass
class Go2PiperMoEEnvCfg(LocomotionVelocityEnvCfg):
    """CTS-MoE flat-terrain task config."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = GO2PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # event
        self.events.push_robot = None

        # flat terrain 
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        ##  velocity command
        self.commands.base_velocity.curriculum_coeff = 4000
        # init
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges_init.lin_vel_x  = (-0.0, 0.3)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (0.0, 0.0)
        # final
        self.commands.base_velocity.ranges_final.lin_vel_x = (-0.0, 0.8)
        self.commands.base_velocity.ranges_final.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges_final.ang_vel_z = (0.0, 0.0)
        # Flat task can use a separate command curriculum while other tasks keep the ranges above.
        self.commands.base_velocity.flat_ranges_init = command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.3),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1),
            heading=(-0.0, 0.0),
        )
        self.commands.base_velocity.flat_ranges_final = command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.8),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.5, 0.5),
            heading=(-0.0, 0.0),
        )
  
        ## position command 
        self.commands.ee_pose.curriculum_coeff = 1
        self.commands.ee_pose.resampling_time_range = (4.0, 6.0)
        # init
        self.commands.ee_pose.ranges_init.pos_x = (0.33, 0.38)
        self.commands.ee_pose.ranges_init.pos_y = (-0.05, 0.05)
        self.commands.ee_pose.ranges_init.pos_z = (0.55, 0.6)
        self.commands.ee_pose.ranges_init.pitch = (0.0, 3.14 / 4)
        # self.commands.ee_pose.ranges_init.pos_x = (-0.6, 0.6)
        # self.commands.ee_pose.ranges_init.pos_y = (-0.5, 0.5)
        # self.commands.ee_pose.ranges_init.pos_z = (0.55, 0.55)

        # final
        self.commands.ee_pose.ranges_final.pos_x = (0.55, 0.55)
        self.commands.ee_pose.ranges_final.pos_y = (0.0, 0.0)
        self.commands.ee_pose.ranges_final.pos_z = (0.4, 0.4)
        
        self.commands.ee_pose.ranges_final.pitch = (0.0, 3.14 / 4)
        self.commands.ee_pose.ranges.pitch = (3.14 / 4, 3.14 / 4)

        
        # Common reward weights.  Reward terms ending with "_common" are used by all tasks.
        self.rewards.end_effector_position_tracking_exp_common.weight = 4.0
        self.rewards.end_effector_position_tracking_l2_common.weight = -0.0
        self.rewards.end_effector_position_tracking_fine_grained_common.weight = 2.0
        self.rewards.end_effector_orientation_tracking_common.weight = -3.0
        self.rewards.end_effector_action_rate_common.weight = -0.005 #-0.005 
        self.rewards.end_effector_action_smoothness_common.weight = -0.02#-0.02
        self.rewards.end_effector_joint_vel_common.weight = -0.001 # -0.0001
        self.rewards.end_effector_lin_vel_z_l2_common.weight = -0.0
        self.rewards.end_effector_ang_vel_xy_l2_common.weight = -0.0        
        self.rewards.end_effector_flat_orientation_l2_common.weight = -0.0
        self.rewards.track_ang_vel_z_exp_common.weight = 4.0
        self.rewards.track_ori_exp_common.weight = 1.0
        self.rewards.ang_vel_xy_l2_common.weight = -0.1
        self.rewards.dof_torques_l2_common.weight = -1.0e-5 
        self.rewards.dof_acc_l2_common.weight =  -2.5e-7
        self.rewards.action_rate_l2_common.weight = -0.01
        self.rewards.feet_air_time_common.weight = 0.4
        self.rewards.feet_slide_common.weight = -0.05
        self.rewards.F_feet_air_time_common.weight = 0.0 #0.5
        self.rewards.R_feet_air_time_common.weight = 0.0 #0.5
        self.rewards.feet_height_common.weight = -0.2 #TODO # -0.2
        self.rewards.feet_height_body_common.weight = -0.0 # 0.5 #TODO 
        self.rewards.foot_contact_common.weight = 0.005
        self.rewards.joint_mirror_common.weight =  -0.05
        self.rewards.gait_reward_common.weight = 1.0 # 1.0
        self.rewards.feet_long_air_common.weight =  -0.1
        self.rewards.hip_deviation_common.weight = -0.2
        self.rewards.joint_deviation_common.weight = -0.00 # 0.0
        self.rewards.F_joint_deviation_common.weight = -0.06 # 0.1
        self.rewards.R_joint_deviation_common.weight = -0.1 # 0.15
        self.rewards.action_smoothness_common.weight = -0.02 # -0.02


        # Box-avoidance reward weights:
        # Add RewTerm fields ending with "_box_avoidance" in RewardsCfg, then configure them here.
        self.rewards.track_lin_vel_x_exp_box_avoidance.weight = 4.0
        self.rewards.track_lin_vel_y_exp_box_avoidance.weight = 0.5
        self.rewards.track_base_height_exp_box_avoidance.weight = 1.0
        self.rewards.lin_vel_z_l2_box_avoidance.weight = -2.5
        self.rewards.thigh_contact_box_avoidance.weight = -1.0
        self.rewards.calf_contact_box_avoidance.weight = -1.0
        self.rewards.base_contact_box_avoidance.weight = -1.0
        self.rewards.arm_contact_box_avoidance.weight = -1.0
        self.rewards.flat_orientation_l2_box_avoidance.weight = -0.5 # -0.5

        # Under-table reward weights:
        # Add RewTerm fields ending with "_under_table" in RewardsCfg, then configure them here.
        self.rewards.track_lin_vel_x_exp_under_table.weight = 4.0
        self.rewards.track_lin_vel_y_exp_under_table.weight = 4.0
        self.rewards.track_base_height_exp_under_table.weight = 1.0
        self.rewards.lin_vel_z_l2_under_table.weight = -1.0
        self.rewards.thigh_contact_under_table.weight = -1.0
        self.rewards.calf_contact_under_table.weight = -1.0
        self.rewards.base_contact_under_table.weight = -1.0
        self.rewards.arm_contact_under_table.weight = -1.0
        self.rewards.flat_orientation_l2_under_table.weight = -0.5 # -0.5

        # Stair-up reward weights:
        # Add RewTerm fields ending with "_stair_up" in RewardsCfg, then configure them here.
        self.rewards.track_lin_vel_x_exp_stair_up.weight = 4.0
        self.rewards.track_lin_vel_y_exp_stair_up.weight = 4.0
        self.rewards.track_base_height_exp_stair_up.weight = 1.0
        self.rewards.lin_vel_z_l2_stair_up.weight = -0.25
        self.rewards.thigh_contact_stair_up.weight = -1.0
        self.rewards.calf_contact_stair_up.weight = -1.0
        self.rewards.base_contact_stair_up.weight = -1.0
        self.rewards.arm_contact_stair_up.weight = -1.0
        self.rewards.flat_orientation_l2_stair_up.weight = -0.0 # -0.5


        # Flat reward weights:
        # Add RewTerm fields ending with "_flat" in RewardsCfg, then configure them here.
        self.rewards.track_lin_vel_x_exp_flat.weight = 4.0
        self.rewards.track_lin_vel_y_exp_flat.weight = 4.0
        self.rewards.track_base_height_exp_flat.weight = 1.0
        self.rewards.lin_vel_z_l2_flat.weight = -2.5
        self.rewards.thigh_contact_flat.weight = -0.5
        self.rewards.calf_contact_flat.weight = -0.5
        self.rewards.base_contact_flat.weight = -0.5
        self.rewards.arm_contact_flat.weight = -0.5
        self.rewards.flat_orientation_l2_flat.weight = -0.5 # -0.5




class Go2PiperMoEEnvCfg_PLAY(Go2PiperMoEEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.proprio.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
        
        # self.commands.ee_pose.debug_vis = False
        # self.commands.base_velocity.debug_vis = False

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # self.events.reset_base = None
        # self.events.reset_base = None
        # self.terminations.base_contact = None
        # self.terminations.calf_contact = None
        # self.terminations.thigh_contact = None


        self.commands.ee_pose.is_Go2ARM = False
        self.commands.base_velocity.is_Go2ARM = False
  
        self.commands.ee_pose.is_Go2ARM_Play = True
        
        self.commands.base_velocity.resampling_time_range = (5.0,5.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        
        # final
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
       
        self.commands.ee_pose.resampling_time_range = (2.5, 3.5) 

        self.commands.ee_pose.ranges.pos_x = (0.4, 0.6)
        self.commands.ee_pose.ranges.pos_y = (-0.1, 0.1)
        self.commands.ee_pose.ranges.pos_z = (0.1, 0.6)

        # self.commands.ee_pose.ranges.pos_x = (0.5, 0.5)
        # self.commands.ee_pose.ranges.pos_y = (-0.0, 0.0)
        # self.commands.ee_pose.ranges.pos_z = (0.5, 0.5)
