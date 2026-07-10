# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, subtract_frame_transforms , quat_conjugate, axis_angle_from_quat
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PLAY = False
import numpy as np

# ================================================================================================================================

def position_command_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    # des_pos_xy_b = command[:, :2]
    # des_pos_z_w = command[:, 2]

    # print("des_pos_xy_b",des_pos_xy_b)
    # print("des_pos_z_w",des_pos_z_w)

    # print("asset.data.root_state_w",asset.data.root_state_w[:, :3])
    # des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    # des_pos_w[:,2] = des_pos_b[:,2] + asset.data.root_state_w[:, 2] # TODO:!!!
    # curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    # print("curr_pos_w",curr_pos_w[:, :3])
    # output = torch.exp(-torch.sum(torch.square(curr_pos_w - des_pos_w) / std, dim=1))
    # print("des_pos_w",des_pos_w[:, :3])
    
    end_effector_curr_pos_b = asset.data.body_pos_w[:, asset_cfg.body_ids[0]] - asset.data.root_pos_w
    end_effector_curr_pos_b = quat_apply_inverse(asset.data.root_state_w[:, 3:7], end_effector_curr_pos_b)  
    ee_pos_xy_err = torch.abs( end_effector_curr_pos_b[:, :2] - command[:, :2] )
    ee_pos_z_err =  torch.abs(command[:, 2:3] - asset.data.body_pos_w[:, asset_cfg.body_ids[0]][:, 2:3]  )
    pos_error = torch.cat([ee_pos_xy_err, ee_pos_z_err], dim=-1)
    output = torch.exp(-torch.sum(torch.square(pos_error) / std, dim=1))

    # print("--reward--")
    # print("command",command)
    # print("end_effector_curr_pos_b",end_effector_curr_pos_b)
    # print("asset.data.body_pos_w[:, asset_cfg.body_ids[0]]",asset.data.body_pos_w[:, asset_cfg.body_ids[0]])
    # print("ee_pos_xy_err",ee_pos_xy_err)
    # print("ee_pos_z_err",ee_pos_z_err)
    # print("pos_error",pos_error)
    # print("----")

    return output

def position_command_error_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    end_effector_curr_pos_b = asset.data.body_pos_w[:, asset_cfg.body_ids[0]] - asset.data.root_pos_w
    end_effector_curr_pos_b = quat_apply_inverse(asset.data.root_state_w[:, 3:7], end_effector_curr_pos_b)  
    ee_pos_xy_err = torch.abs( end_effector_curr_pos_b[:, :2] - command[:, :2] )
    ee_pos_z_err =  torch.abs(command[:, 2:3] - asset.data.body_pos_w[:, asset_cfg.body_ids[0]][:, 2:3]  )
    pos_error = torch.cat([ee_pos_xy_err, ee_pos_z_err], dim=-1)

    return torch.norm(pos_error, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    end_effector_curr_pos_b = asset.data.body_pos_w[:, asset_cfg.body_ids[0]] - asset.data.root_pos_w
    end_effector_curr_pos_b = quat_apply_inverse(asset.data.root_state_w[:, 3:7], end_effector_curr_pos_b)  
    ee_pos_xy_err = torch.abs( end_effector_curr_pos_b[:, :2] - command[:, :2] )
    ee_pos_z_err =  torch.abs(command[:, 2:3] - asset.data.body_pos_w[:, asset_cfg.body_ids[0]][:, 2:3]  )
    pos_error = torch.cat([ee_pos_xy_err, ee_pos_z_err], dim=-1)
    distance = torch.norm(pos_error, dim=1)
    # print("distance",distance)
    # print("output",1 - torch.tanh(distance / std))
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    source_quat_norm = quat_mul(des_quat_w, quat_conjugate(des_quat_w))[:, 0]
    source_quat_inv = quat_conjugate(des_quat_w) / source_quat_norm.unsqueeze(-1)
    quat_error = quat_mul(curr_quat_w, source_quat_inv) 
    rot_error = axis_angle_from_quat(quat_error)
    # des_rot = axis_angle_from_quat(des_quat_w)
    # curr_rot = axis_angle_from_quat(curr_quat_w)
    # print("des_rot",des_rot)
    # print("curr_rot",curr_rot)
    # print("rot_error",rot_error)
    return torch.norm(rot_error, dim=-1)



def base_ori_tracking(    
    env: ManagerBasedRLEnv,command_name: str,std: float,
):
    base_ori = env.scene["robot"].data.projected_gravity_b[:, :2]
    command = env.command_manager.get_command(command_name)# shape: (N,)
    des_ori_y = torch.clamp(command[:,1], min=-0.1, max=0.1)
    des_ori_x = torch.clamp(-(command[:,2] - 0.45), min=-0.1, max=0.15)
    
    des_ori = torch.cat([des_ori_x.unsqueeze(1), des_ori_y.unsqueeze(1)], dim=1) 
    ori_error = torch.sum(
        torch.abs(base_ori - des_ori),
        dim=1,
    )
    return torch.exp(-ori_error / std)



def robot_in_table_xy_region(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    table_object_name: str = "table_top",
    table_half_extents_xy: tuple[float, float] = (1.0, 0.7),
    min_table_height_w: float = 0.1,
) -> torch.Tensor:
    """Return 1.0 when the robot base is inside the active table footprint, else 0.0."""
    robot: RigidObject = env.scene[asset_cfg.name]
    table: RigidObject = env.scene[table_object_name]

    robot_xy = robot.data.root_pos_w[:, :2]
    table_xy = table.data.root_pos_w[:, :2]
    table_active = table.data.root_pos_w[:, 2] > min_table_height_w

    offset = robot_xy - table_xy
    half_x, half_y = table_half_extents_xy
    in_region = (
        table_active
        & (torch.abs(offset[:, 0]) <= half_x)
        & (torch.abs(offset[:, 1]) <= half_y)
    )
    return in_region.float()


def base_height_tracking(
    env: ManagerBasedRLEnv,
    desired_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    terrain_height_mode: str = "max",
):
    asset: RigidObject = env.scene[asset_cfg.name]

    base_height_w = asset.data.root_pos_w[:, 2]
    sensor = env.scene.sensors[sensor_cfg.name]
    terrain_heights_w = sensor.data.ray_hits_w[..., 2]
    valid_hits = torch.isfinite(terrain_heights_w)

    if terrain_height_mode == "mean":
        safe_heights = torch.where(valid_hits, terrain_heights_w, torch.zeros_like(terrain_heights_w))
        valid_counts = valid_hits.sum(dim=1).clamp(min=1)
        local_terrain_height_w = safe_heights.sum(dim=1) / valid_counts
    elif terrain_height_mode == "max":
        safe_heights = torch.where(valid_hits, terrain_heights_w, torch.full_like(terrain_heights_w, -torch.inf))
        local_terrain_height_w = torch.max(safe_heights, dim=1).values
        local_terrain_height_w = torch.where(torch.isfinite(local_terrain_height_w), local_terrain_height_w, torch.zeros_like(local_terrain_height_w))
    else:
        raise ValueError(f"Unsupported terrain_height_mode: {terrain_height_mode}")

    height_above_terrain = base_height_w - local_terrain_height_w
    height_error = torch.abs(desired_height - height_above_terrain)

    return torch.exp(-height_error / std)


def base_height_tracking_in_table_region(
    env: ManagerBasedRLEnv,
    desired_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    terrain_height_mode: str = "max",
    table_object_name: str = "table_top",
    table_half_extents_xy: tuple[float, float] = (1.0, 0.7),
    min_table_height_w: float = 0.1,
) -> torch.Tensor:
    """Track base height above terrain only after the robot enters the table footprint."""
    reward = base_height_tracking(
        env,
        desired_height=desired_height,
        std=std,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        terrain_height_mode=terrain_height_mode,
    )
    gate = robot_in_table_xy_region(
        env,
        asset_cfg=asset_cfg,
        table_object_name=table_object_name,
        table_half_extents_xy=table_half_extents_xy,
        min_table_height_w=min_table_height_w,
    )
    return reward * gate


def action_rate_l2_arm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:,12:] - env.action_manager.prev_action[:,12:]), dim=1)

def arm_action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action[:, 12:] - env.action_manager.prev_action[:, 12:]), dim=1)

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    
    lin_vel_error = torch.sum(
        torch.abs(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std)


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of x-axis linear velocity command using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.abs(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-lin_vel_error / std)


def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of y-axis linear velocity command using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.abs(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])
    return torch.exp(-lin_vel_error / std)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.abs(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def joint_torques_l2_Go2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    leg_joint, _ = asset.find_joints([ "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
                        ])
    return torch.sum(torch.square(asset.data.applied_torque[:, leg_joint]), dim=1)


def joint_acc_l2_Go2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    leg_joint, _ = asset.find_joints([ "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
                        ])
    return torch.sum(torch.square(asset.data.joint_acc[:, leg_joint]), dim=1)


def action_rate_l2_Go2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:,:12] - env.action_manager.prev_action[:,:12]), dim=1)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
    
def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


    return reward



def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)

    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])

    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def standing_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str,
                                force_threshold: float, command_threshold: float) -> torch.Tensor:
    # Extract the relevant sensor and command
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)  # shape: (N, B)
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)  # shape: (N,)

    # Check conditions
    is_small_command = command < command_threshold

    force = torch.min(contact_force, dim=1).values  
    force = torch.clamp(force,min=0.0,max =force_threshold)
    rewards = torch.where(is_small_command, 
                          2.0* (force), 
                          force)
    rewards *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return rewards



def feet_long_air_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_time: float = 0.5,
) -> torch.Tensor:
    """
    Penalize feet that stay in the air longer than max_air_time.
    Penalty = Σ( clamp(current_air_time - max_air_time, min=0) )^2

    Args:
        env: RL environment.
        sensor_cfg: Must point to the ContactSensor.
        max_air_time: Threshold above which penalty starts (s).
        scale: Overall penalty strength (positive).

    Returns:
        Tensor of shape (num_envs,)  **<= 0**  (negative reward).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # 如果传感器没开 air_time 跟踪，直接返回 0
    if contact_sensor.data.current_air_time is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 只考虑被跟踪的脚
    current_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # (N, n_feet)

    # 超限部分
    excess = torch.clamp(current_air - max_air_time, min=0.0)  # < 0.5 时 = 0

    # 平方惩罚并求和
    penalty = torch.sum(torch.square(excess), dim=1)  # (N,)
    return penalty

def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def hip_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, [0, 3 , 6, 9]]), dim=1)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)



def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    # print("last_ari_time",last_air_time)
    # print("current_air_time",current_air_time)
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    # print("last_contact_time",last_contact_time)
    # print("current_contact_time",current_contact_time)
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + \
             torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    # print("rew",reward)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# ================================================================================================================================


# def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking of the position error using L2-norm.

#     The function computes the position error between the desired position (from the command) and the
#     current position of the asset's body (in world frame). The position error is computed as the L2-norm
#     of the difference between the desired and current positions.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current positions
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
#     curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
#     return torch.norm(curr_pos_w - des_pos_w, dim=1)


# def position_command_error_tanh(
#     env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#     """Reward tracking of the position using the tanh kernel.

#     The function computes the position error between the desired position (from the command) and the
#     current position of the asset's body (in world frame) and maps it with a tanh kernel.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current positions
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
#     curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
#     distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
#     return 1 - torch.tanh(distance / std)


# def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize tracking orientation error using shortest path.

#     The function computes the orientation error between the desired orientation (from the command) and the
#     current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
#     path between the desired and current orientations.
#     """
#     # extract the asset (to enable type hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # obtain the desired and current orientations
#     des_quat_b = command[:, 3:7]
#     des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
#     curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
#     return quat_error_magnitude(curr_quat_w, des_quat_w)




def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""





def base_height_l2(
    env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.05,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # TODO: Fix this for rough-terrain.
    curr_height = torch.clamp(asset.data.root_pos_w[:, 2], max=0.4)
    return torch.square((curr_height - target_height)/std)

def base_height_exp(
    env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.05,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # TODO: Fix this for rough-terrain.
    curr_height = torch.clamp(asset.data.root_pos_w[:, 2], max=0.4)
    err = torch.abs(curr_height - target_height)
    output = torch.exp(err / std)
    return output


def probe_links_below_height_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_height: float = 0.5,
    std: float = 0.05,
) -> torch.Tensor:
    """Reward keeping probe links below a world-frame height threshold.

    Uses the maximum probe height so all listed links must stay under ``max_height``
    to receive full reward. Reward decays exponentially as any probe exceeds the limit.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    probe_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    max_probe_z = probe_heights.max(dim=1).values
    excess = torch.clamp(max_probe_z - max_height, min=0.0)
    return torch.exp(-excess / std)


def probe_links_below_height_exp_in_table_region(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_height: float = 0.5,
    std: float = 0.05,
    table_object_name: str = "table_top",
    table_half_extents_xy: tuple[float, float] = (1.0, 0.7),
    min_table_height_w: float = 0.1,
) -> torch.Tensor:
    """Reward probe clearance only after the robot enters the table footprint."""
    reward = probe_links_below_height_exp(
        env,
        asset_cfg=asset_cfg,
        max_height=max_height,
        std=std,
    )
    gate = robot_in_table_xy_region(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        table_object_name=table_object_name,
        table_half_extents_xy=table_half_extents_xy,
        min_table_height_w=min_table_height_w,
    )
    return reward * gate


    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)




def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)



def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

def joint_vel_l2_arm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    arm_joint, _ = asset.find_joints([ 
                                "joint1"       , "joint2"      , "joint3"        , 
                                "joint4", "joint5"   , "joint6"
                        ])
    return torch.sum(torch.square(asset.data.joint_vel[:, arm_joint]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

def leg_action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action[:, :12] - env.action_manager.prev_action[:, :12]), dim=1)



def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)

##Go2ARM
def joint_arm_energy_abs_sum(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    arm_joint, _ = asset.find_joints([ 
                                "joint1"       , "joint2"      , "joint3"        , 
                                "joint4", "joint5"   , "joint6"
                        ])
    return torch.sum(torch.abs(asset.data.applied_torque[:,arm_joint] * asset.data.joint_vel[:, arm_joint]), dim=1)

##Go2ARM
def joint_leg_energy_abs_sum(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    leg_joint, _ = asset.find_joints([ "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
                        ])
    return torch.sum(torch.abs(asset.data.applied_torque[:, leg_joint] * asset.data.joint_vel[:, leg_joint]), dim=1)


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)



def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)

##Go2ARM



"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)


##Go2ARM
def contact_forces_z(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize contact forces specifically for the z-axis if the net contact force exceeds a threshold.
    """
    # Extract the contact sensor data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history  # [batch_size, time_steps, body_parts, 3]

    # Extract the z-axis contact force
    z_contact_forces = net_contact_forces[:, :, sensor_cfg.body_ids, 2]  # z-axis is the third dimension

    # Compute the violation (force exceeding the threshold)
    violation = torch.max(z_contact_forces, dim=1)[0] - threshold

    # Compute the penalty (sum of violations)
    return torch.sum(violation.clip(min=0.0), dim=1)

"""
Velocity-tracking rewards.
"""











def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
