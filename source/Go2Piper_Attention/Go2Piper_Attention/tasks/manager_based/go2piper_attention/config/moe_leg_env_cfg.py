# Copyright (c) 2022-2024, The Isaac Attention Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from Go2Piper_Attention.tasks.manager_based.go2piper_attention.go2piper_leg_cts_moe_env_cfg import (
    LocomotionVelocityEnvCfg,
)
from Go2Piper_Attention.assets.go2arm_articulation_cfg import GO2PIPER_CFG


@configclass
class Go2PiperMoEEnvCfg(LocomotionVelocityEnvCfg):
    """CTS-MoE flat-terrain task config."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = GO2PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        }
        self.scene.robot.actuators = {"base_legs": GO2PIPER_CFG.actuators["base_legs"]}

        # event
        self.events.push_robot = None

        # flat terrain 
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        ##  velocity command
        self.commands.base_velocity.curriculum_coeff = 4000
        # init
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.ranges_init.lin_vel_x  = (0.0, 0.3)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (-0.1, 0.1)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (-0.1, 0.1)
        # final
        self.commands.base_velocity.ranges_final.lin_vel_x = (0.3, 0.8)
        self.commands.base_velocity.ranges_final.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges_final.ang_vel_z = (-0.5, 0.5)
        
        # Common reward weights.  Reward terms ending with "_common" are used by all tasks.
        self.rewards.track_ang_vel_z_exp_common.weight = 2.0
        self.rewards.ang_vel_xy_l2_common.weight = -0.1
        self.rewards.dof_torques_l2_common.weight = -1.0e-5 
        self.rewards.dof_acc_l2_common.weight =  -2.5e-7
        self.rewards.action_rate_l2_common.weight = -0.01
        self.rewards.feet_air_time_common.weight = 0.4
        self.rewards.feet_slide_common.weight = -0.05
        self.rewards.F_feet_air_time_common.weight = 0.0 #0.5
        self.rewards.R_feet_air_time_common.weight = 0.0 #0.5
        # self.rewards.feet_height_common.weight = -0.2 #TODO # -0.2
        # self.rewards.feet_height_body_common.weight = -0.0 # 0.5 #TODO 
        self.rewards.foot_contact_common.weight = 0.005
        self.rewards.joint_mirror_common.weight =  -0.05
        self.rewards.gait_reward_common.weight = 1.0 # 1.0
        self.rewards.feet_long_air_common.weight =  -0.1
        self.rewards.hip_deviation_common.weight = -0.2
        self.rewards.joint_deviation_common.weight = -0.00 # 0.0
        self.rewards.F_joint_deviation_common.weight = -0.06 # 0.1
        self.rewards.R_joint_deviation_common.weight = -0.1 # 0.15
        self.rewards.action_smoothness_common.weight = -0.02 # -0.02


        # Rough reward weights:
        # Add RewTerm fields ending with "_rough" in RewardsCfg, then configure them here.
        self.rewards.track_lin_vel_x_exp_rough.weight = 4.0
        self.rewards.track_lin_vel_y_exp_rough.weight = 4.0
        self.rewards.track_base_height_exp_rough.weight = 1.0
        self.rewards.lin_vel_z_l2_rough.weight = -2.5
        self.rewards.thigh_contact_rough.weight = -1.0
        self.rewards.calf_contact_rough.weight = -1.0
        self.rewards.base_contact_rough.weight = -1.0
        self.rewards.flat_orientation_l2_rough.weight = -0.5 # -0.5
        self.rewards.feet_height_rough.weight = -0.2

        # Floating-ring reward weights:
        # Add RewTerm fields ending with "_floating_ring" in RewardsCfg, then configure them here.
        self.rewards.track_lin_vel_x_exp_floating_ring.weight = 4.0
        self.rewards.track_lin_vel_y_exp_floating_ring.weight = 4.0
        # self.rewards.track_base_height_exp_floating_ring.weight = 0.0
        self.rewards.track_base_height_exp_floating_ring.weight = 2.0
        self.rewards.lin_vel_z_l2_floating_ring.weight = -1.0
        self.rewards.thigh_contact_floating_ring.weight = -1.0
        self.rewards.calf_contact_floating_ring.weight = -1.0
        self.rewards.base_contact_floating_ring.weight = -1.0
        self.rewards.flat_orientation_l2_floating_ring.weight = -1.5 # -0.5
        self.rewards.feet_height_floating_ring.weight = -0.2

        # Ascend reward weights:
        # Add RewTerm fields ending with "_ascend" in RewardsCfg, then configure them here.
        self.rewards.track_lin_vel_x_exp_ascend.weight = 4.0
        self.rewards.track_lin_vel_y_exp_ascend.weight = 4.0
        self.rewards.track_base_height_exp_ascend.weight = 1.0
        self.rewards.lin_vel_z_l2_ascend.weight = -0.25
        self.rewards.thigh_contact_ascend.weight = -0.5
        self.rewards.calf_contact_ascend.weight = -0.5
        self.rewards.base_contact_ascend.weight = -0.5
        self.rewards.flat_orientation_l2_ascend.weight = -0.5 # -0.5
        self.rewards.feet_height_body_ascend.weight = -5.0

        # Descend reward weights:
        # Add RewTerm fields ending with "_descend" in RewardsCfg, then configure them here.
        self.rewards.track_lin_vel_x_exp_descend.weight = 4.0
        self.rewards.track_lin_vel_y_exp_descend.weight = 4.0
        self.rewards.track_base_height_exp_descend.weight = 1.0
        self.rewards.lin_vel_z_l2_descend.weight = -0.25
        self.rewards.thigh_contact_descend.weight = -0.5
        self.rewards.calf_contact_descend.weight = -0.5
        self.rewards.base_contact_descend.weight = -0.5
        self.rewards.flat_orientation_l2_descend.weight = -0.5
        self.rewards.feet_height_body_descend.weight = -5.0


        # Flat reward weights:
        # Add RewTerm fields ending with "_flat" in RewardsCfg, then configure them here.
        self.rewards.track_lin_vel_x_exp_flat.weight = 4.0
        self.rewards.track_lin_vel_y_exp_flat.weight = 4.0
        self.rewards.track_base_height_exp_flat.weight = 1.0
        self.rewards.lin_vel_z_l2_flat.weight = -2.5
        self.rewards.thigh_contact_flat.weight = -0.5
        self.rewards.calf_contact_flat.weight = -0.5
        self.rewards.base_contact_flat.weight = -0.5
        self.rewards.flat_orientation_l2_flat.weight = -1.5 # -0.5
        self.rewards.feet_height_flat.weight = -0.2




class Go2PiperMoEEnvCfg_PLAY(Go2PiperMoEEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 10.0
        # disable randomization for play
        self.observations.proprio.enable_corruption = False
        self.observations.proprio_history.enable_corruption = False
        
        # self.commands.base_velocity.debug_vis = False

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # self.events.reset_base = None
        # self.events.reset_base = None
        # self.terminations.base_contact = None
        # self.terminations.calf_contact = None
        # self.terminations.thigh_contact = None


        self.commands.base_velocity.is_Go2ARM = False
        
        self.commands.base_velocity.resampling_time_range = (1e6, 1e6)
        self.commands.base_velocity.rel_standing_envs = 0.1
        
        # final
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
