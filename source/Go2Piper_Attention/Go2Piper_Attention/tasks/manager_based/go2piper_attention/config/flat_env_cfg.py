# Copyright (c) 2022-2024, The Isaac Attention Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from Go2Piper_Attention.tasks.manager_based.go2piper_attention.go2piper_attention_env_cfg import LocomotionVelocityEnvCfg
from Go2Piper_Attention.assets.go2arm_articulation_cfg import GO2PIPER_CFG


@configclass
class Go2PiperFlatEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = GO2PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # event
        self.events.push_robot = None

        # observation
        # hist
        self.observations.leg_policy.base_ang_vel.history_length = 0
        self.observations.leg_policy.joint_pos.history_length = 0
        self.observations.leg_policy.joint_vel.history_length = 0
        self.observations.leg_policy.actions.history_length = 0
        self.observations.leg_policy.velocity_commands.history_length = 0
        self.observations.leg_policy.projected_gravity.history_length = 0
        # self.observations.leg_policy.pos_commands = None
    
        self.observations.arm_policy.joint_pos.history_length = 0
        self.observations.arm_policy.joint_vel.history_length = 0
        self.observations.arm_policy.actions.history_length = 0
        self.observations.arm_policy.pos_commands.history_length = 0


        # flat terrain 
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        ##  velocity command
        self.commands.base_velocity.curriculum_coeff = 4000
        # init
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges_init.lin_vel_x  = (-0.0, 0.3)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (-0.1, 0.1)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (-0.1, 0.1)
        # final
        self.commands.base_velocity.ranges_final.lin_vel_x = (-0.0, 0.8)
        self.commands.base_velocity.ranges_final.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges_final.ang_vel_z = (-0.5, 0.5)
  
        ## position command 
        self.commands.ee_pose.curriculum_coeff = 4000
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
        self.commands.ee_pose.ranges_final.pos_x = (0.45, 0.65)
        self.commands.ee_pose.ranges_final.pos_y = (-0.2, 0.2)
        self.commands.ee_pose.ranges_final.pos_z = (0.1, 0.6)
        
        self.commands.ee_pose.ranges_final.pitch = (0.0, 3.14 / 4)
        self.commands.ee_pose.ranges.pitch = (3.14 / 4, 3.14 / 4)

        
        # reward weight
        # arm
        self.rewards.end_effector_position_tracking_exp.weight = 4.0
        self.rewards.end_effector_position_tracking_l2.weight = -0.0

        self.rewards.end_effector_position_tracking_fine_grained.weight = 2.0
        self.rewards.end_effector_orientation_tracking.weight = -3.0
        # self.rewards.end_effector_action_rate.weight = -0.01 #-0.005 
        # self.rewards.end_effector_action_smoothness.weight = -0.02 #-0.02
        self.rewards.end_effector_action_rate.weight = -0.005 #-0.005 
        self.rewards.end_effector_action_smoothness.weight = -0.02#-0.02
        self.rewards.end_effector_joint_vel.weight = -0.001 # -0.0001
        self.rewards.end_effector_lin_vel_z_l2.weight = -0.0
        self.rewards.end_effector_ang_vel_xy_l2.weight = -0.0        
        self.rewards.end_effector_flat_orientation_l2.weight = -0.0

        # leg     
        self.rewards.tracking_lin_vel_x_l1.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 4.0
        self.rewards.track_ori_exp.weight = 1.0
        self.rewards.track_base_height_exp.weight = 1.0

        self.rewards.lin_vel_z_l2.weight = -2.5
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.dof_torques_l2.weight = -1.0e-5 
        self.rewards.dof_acc_l2.weight =  -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        
        self.rewards.feet_air_time.weight = 0.4
        self.rewards.feet_slide.weight = -0.05
        
        self.rewards.F_feet_air_time.weight = 0.0 #0.5
        self.rewards.R_feet_air_time.weight = 0.0 #0.5

        self.rewards.feet_height.weight = -0.2 #TODO # -0.2
        self.rewards.feet_height_body.weight = -0.0 # 0.5 #TODO 

        self.rewards.foot_contact.weight = 0.005
        self.rewards.joint_mirror.weight =  -0.05
        self.rewards.gait_reward.weight = 1.0 # 1.0
        self.rewards.feet_long_air.weight =  -0.1
        self.rewards.hip_deviation.weight = -0.2
        self.rewards.joint_deviation.weight = -0.00 # 0.0
        self.rewards.F_joint_deviation.weight = -0.06 # 0.1
        self.rewards.R_joint_deviation.weight = -0.1 # 0.15
        self.rewards.action_smoothness.weight = -0.02 # -0.02
        self.rewards.flat_orientation_l2.weight = -0.5 # -0.5

        # self.rewards.lin_vel_z_l2.weight = -3.0 # -2.5
        # self.rewards.ang_vel_xy_l2.weight = -0.15 # -0.1
        # self.rewards.dof_torques_l2.weight = -2.0e-5 # -1.0e-5 
        # self.rewards.dof_acc_l2.weight = -3e-7 # -2.5e-7
        # self.rewards.action_rate_l2.weight = -0.015 # -0.01
        
        # self.rewards.feet_air_time.weight = 0.3 #0.25
        # self.rewards.feet_slide.weight = -0.08 # -0.05
        
        # self.rewards.F_feet_air_time.weight = 0.0 #0.5
        # self.rewards.R_feet_air_time.weight = 0.0 #0.5

        # self.rewards.feet_height.weight = -0.2 #TODO # -0.2
        # self.rewards.feet_height_body.weight = -0.0 # 0.5 #TODO 

        # self.rewards.foot_contact.weight = 0.008 #0.005
        # self.rewards.joint_mirror.weight = -0.08 # -0.05
        # self.rewards.gait_reward.weight = 1.0 # 1.0
        # self.rewards.feet_long_air.weight = -0.5 # -0.1
        # self.rewards.hip_deviation.weight = -0.2 # -0.2
        # self.rewards.joint_deviation.weight = -0.00 # 0.0
        # self.rewards.F_joint_deviation.weight = -0.1 # 0.1
        # self.rewards.R_joint_deviation.weight = -0.15 # 0.15
        # self.rewards.action_smoothness.weight = -0.03 # -0.02
        # self.rewards.flat_orientation_l2.weight = -0.5 # -0.5


class Go2PiperFlatEnvCfg_PLAY(Go2PiperFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.leg_policy.enable_corruption = False
        self.observations.arm_policy.enable_corruption = False
        
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
