
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, MultiMeshRayCasterCfg, MultiMeshRayCasterCameraCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.terrains as terrain_gen


import Go2Piper_Attention.tasks.manager_based.go2piper_attention.mdp as mdp
from Go2Piper_Attention.tasks.manager_based.go2piper_attention.mdp import leg_observations as leg_obs
from Go2Piper_Attention.tasks.manager_based.go2piper_attention.mdp import arm_observations as arm_obs
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


def cts_moe_task_context(env) -> torch.Tensor:
    """Task context scalar ct as [B, 1], using task ids 0..3."""
    if not hasattr(env, "task_id"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return env.task_id.float().unsqueeze(-1)


##
# Scene definition
##

GO2ARM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.4),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.6, noise_range=(-0.05, 0.05), noise_step=0.01, border_width=0.25
        ),
    },
)


# Kinematic task-scene obstacles: mesh colliders with a kinematic rigid body.
# ManagerRLEnv repositions them via write_root_state_to_sim() so PhysX collision
# stays in sync with the robot.
STATIC_OBSTACLE_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
)
STATIC_OBSTACLE_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(
    collision_enabled=True,
)
STATIC_OBSTACLE_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2ARM_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.1, 0.0, 3.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.3]),
        debug_vis=False,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/stair_step_*", track_mesh_transforms=True),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/stair_platform", track_mesh_transforms=True),
        ],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


    H_ground_scan = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/stair_step_*", track_mesh_transforms=True),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/stair_platform", track_mesh_transforms=True),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/box_obstacle", track_mesh_transforms=True),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/table_top", track_mesh_transforms=True),
        ],
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 3.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        # debug_vis=True,
        debug_vis=True,
    )

    H_lateral_scan = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/stair_step_*", track_mesh_transforms=True),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/stair_platform", track_mesh_transforms=True),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/box_obstacle", track_mesh_transforms=True),
        ],
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 3.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        # debug_vis=True,
        debug_vis=False,
    )

    H_overhead_scan = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/stair_step_*", track_mesh_transforms=True),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/stair_platform", track_mesh_transforms=True),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/table_top", track_mesh_transforms=True),
        ],
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 3.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        # debug_vis=True,
        debug_vis=False,
    )

    depth_image = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        # RealSense D435/D435i 常用 30 FPS
        # update_period 单位是秒，1/30 ≈ 0.0333
        update_period=1.0 / 30.0,
        # 只保留当前帧
        history_length=0,
        debug_vis=False,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/stair_step_*", track_mesh_transforms=True
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/stair_platform", track_mesh_transforms=True
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/table_top", track_mesh_transforms=True),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/table_leg_*", track_mesh_transforms=True),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/box_obstacle", track_mesh_transforms=True
            ),
        ],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            # 保留你原来的 focal_length 数值，只重新计算 aperture 来匹配 87° × 58°
            # horizontal_aperture = 2 * f * tan(87° / 2) = 3.4353
            # vertical_aperture   = 2 * f * tan(58° / 2) = 2.0066
            focal_length=1.81,
            horizontal_aperture=3.4353,
            vertical_aperture=2.0066,
            horizontal_aperture_offset=0.0,
            vertical_aperture_offset=0.0,
            # 训练/策略输入使用降采样后的尺寸
            width=48,
            height=27,
        ),
        data_types=["distance_to_image_plane"],
        # D435 官方工作距离约 0.3m ~ 3m；
        # 你如果只看近距离障碍物，2.0 可以；
        # 如果想更接近真实相机范围，可以改成 3.0
        max_distance=2.0,
        depth_clipping_behavior="max",
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.32715, -0.00003, 0.09),
            rot=(0.3535533906, -0.6123724357, 0.6123724357, -0.3535533906),  # quaternion: w, x, y, z
        ),
    )

    # Task-specific kinematic scene objects. They are spawned once for every env and then
    # moved by ManagerRLEnv via write_root_state_to_sim(). Runtime scale is avoided
    # because MultiMeshRayCaster caches mesh vertices at initialization.
    box_obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box_obstacle",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5, 0.65),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.08), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )

    table_top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table_top",
        spawn=sim_utils.MeshCuboidCfg(
            size=(2.0, 1.4, 0.08),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.30, 0.85), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    table_leg_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table_leg_0",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.06, 0.06, 0.55),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.30, 0.85), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    table_leg_1 = table_leg_0.replace(prim_path="{ENV_REGEX_NS}/table_leg_1")
    table_leg_2 = table_leg_0.replace(prim_path="{ENV_REGEX_NS}/table_leg_2")
    table_leg_3 = table_leg_0.replace(prim_path="{ENV_REGEX_NS}/table_leg_3")

    stair_step_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stair_step_0",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.35, 3.0, 0.05),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    stair_step_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stair_step_1",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.35, 3.0, 0.1),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    stair_step_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stair_step_2",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.35, 3.0, 0.15),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    stair_step_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stair_step_3",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.35, 3.0, 0.2),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    stair_step_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stair_step_4",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.35, 3.0, 0.25),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    stair_step_5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stair_step_5",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.35, 3.0, 0.3),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    stair_step_6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stair_step_6",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.35, 3.0, 0.35),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    stair_platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stair_platform",
        spawn=sim_utils.MeshCuboidCfg(
            size=(5.0, 3.0, 0.35),
            rigid_props=STATIC_OBSTACLE_RIGID_PROPS,
            collision_props=STATIC_OBSTACLE_COLLISION_PROPS,
            physics_material=STATIC_OBSTACLE_PHYSICS_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )

                    

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # add_ee_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
    #         "mass_distribution_params": (-0.1, 0.5),
    #         "operation": "add",
    #     },
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    

    # randomize_rigid_body_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia, ## TODO : 
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    ## Go2ARM
    
    ee_pose = mdp.command_cfg.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="end_effector",
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        is_Go2ARM=True,
        curriculum_coeff = 1000,          
        ranges_final =mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.35, 0.35),
            pos_z=(0.1, 0.55), # world frame not base frame

            roll=(-0.0, 0.0),
            pitch=(-0.0, -0.0),  # depends on end-effector axis
            yaw=(-0.0, -0.0),
        ),
        ranges = mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.35, 0.35),
            pos_z=(0.1, 0.55), # world frame not base frame

            roll=(-0.0, 0.0),
            pitch=(-0.0, -0.0),  # depends on end-effector axis
            yaw=(-0.0, -0.0),
        ),
        ranges_init=mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.5), 
            pos_y=(-0.05, 0.05),
            pos_z=(0.35, 0.4), # world frame not base frame

            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),  # depends on end-effector axis
            yaw=(-0.0, 0.0),
        ),
    )

    base_velocity = mdp.command_cfg.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        is_Go2ARM=True,
        curriculum_coeff= 1000,         
        ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-0.0, 0.0)
        ),
        ranges_final=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.8), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-0.0, 0.0)
        ),
        ranges_init=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.3), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=[
                                                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                                    ],
                                        scale=0.25,
                                         use_default_offset=True,
                                         preserve_order=True,
    )   
    arm_pose = mdp.JointPositionActionCfg(asset_name="robot",
                                          joint_names=[
                                              "joint1", "joint2", "joint3", 
                                              "joint4", "joint5", "joint6"],
                                            scale=0.5,
                                            use_default_offset=True,
                                            preserve_order=True,)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""     

    @configclass
    class ProprioCfg(ObsGroup):
        """Current full-body proprioception for CTS-MoE actor and critic."""

        base_ang_vel = ObsTerm(func=leg_obs.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # dim = 3
        leg_joint_pos = ObsTerm(func=leg_obs.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))  # dim = 12
        leg_joint_vel = ObsTerm(func=leg_obs.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))  # dim = 12
        leg_actions = ObsTerm(func=leg_obs.last_action, params={"action_name": "joint_pos"})  # dim = 12
        arm_joint_pos = ObsTerm(func=arm_obs.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))  # dim = 6
        arm_joint_vel = ObsTerm(func=arm_obs.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))  # dim = 6
        arm_actions = ObsTerm(func=arm_obs.last_action, params={"action_name": "arm_pose"})  # dim = 6
        velocity_commands = ObsTerm(
            func=leg_obs.generated_commands,
            params={"command_name": "base_velocity"},
        )  # dim = 3
        ee_pose_commands = ObsTerm(
            func=arm_obs.generated_commands,
            params={"command_name": "ee_pose"},
        )  # dim = 3
        projected_gravity = ObsTerm(
            func=leg_obs.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )  # dim = 3
        ct = ObsTerm(func=cts_moe_task_context)  # dim = 1

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioHistoryCfg(ObsGroup):
        """Five-frame full-body proprioception history for the student encoder."""

        base_ang_vel = ObsTerm(func=leg_obs.base_ang_vel, history_length=5, noise=Unoise(n_min=-0.2, n_max=0.2))  # dim = 3
        leg_joint_pos = ObsTerm(func=leg_obs.joint_pos_rel, history_length=5, noise=Unoise(n_min=-0.05, n_max=0.05))  # dim = 12
        leg_joint_vel = ObsTerm(func=leg_obs.joint_vel_rel, history_length=5, noise=Unoise(n_min=-1.5, n_max=1.5))  # dim = 12
        leg_actions = ObsTerm(func=leg_obs.last_action, history_length=5, params={"action_name": "joint_pos"})  # dim = 12
        arm_joint_pos = ObsTerm(func=arm_obs.joint_pos_rel, history_length=5, noise=Unoise(n_min=-0.05, n_max=0.05))  # dim = 6
        arm_joint_vel = ObsTerm(func=arm_obs.joint_vel_rel, history_length=5, noise=Unoise(n_min=-1.5, n_max=1.5))  # dim = 6
        arm_actions = ObsTerm(func=arm_obs.last_action, history_length=5, params={"action_name": "arm_pose"})  # dim = 6
        velocity_commands = ObsTerm(
            func=leg_obs.generated_commands,
            history_length=5,
            params={"command_name": "base_velocity"},
        )  # dim = 3
        ee_pose_commands = ObsTerm(
            func=arm_obs.generated_commands,
            history_length=5,
            params={"command_name": "ee_pose"},
        )  # dim = 3
        projected_gravity = ObsTerm(
            func=leg_obs.projected_gravity,
            history_length=5,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )  # dim = 3
        ct = ObsTerm(func=cts_moe_task_context, history_length=5)  # dim = 1

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedObsCfg(ObsGroup):
        """Full-body privileged observations for CTS-MoE teacher encoder."""

        base_ang_vel = ObsTerm(func=leg_obs.base_ang_vel)  # dim = 3
        base_lin_vel = ObsTerm(func=leg_obs.base_lin_vel)  # dim = 3
        leg_joint_pos = ObsTerm(func=leg_obs.joint_pos_rel)  # dim = 12
        leg_joint_vel = ObsTerm(func=leg_obs.joint_vel_rel)  # dim = 12
        leg_actions = ObsTerm(func=leg_obs.last_action, params={"action_name": "joint_pos"})  # dim = 12
        arm_joint_pos = ObsTerm(func=arm_obs.joint_pos_rel)  # dim = 6
        arm_joint_vel = ObsTerm(func=arm_obs.joint_vel_rel)  # dim = 6
        arm_actions = ObsTerm(func=arm_obs.last_action, params={"action_name": "arm_pose"})  # dim = 6
        velocity_commands = ObsTerm(
            func=leg_obs.generated_commands,
            params={"command_name": "base_velocity"},
        )  # dim = 3
        ee_pose_commands = ObsTerm(
            func=arm_obs.generated_commands,
            params={"command_name": "ee_pose"},
        )  # dim = 3
        projected_gravity = ObsTerm(func=leg_obs.projected_gravity)  # dim = 3
        leg_joint_torques = ObsTerm(func=leg_obs.get_joints_torques)  # dim = 12
        arm_joint_torques = ObsTerm(func=arm_obs.get_joints_torques)  # dim = 6
        feet_contact = ObsTerm(
            func=leg_obs.feet_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )  # dim = 4 bool
        gripper_pose = ObsTerm(
            func=arm_obs.end_effector_pos_ori_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="end_effector")},
        )  # dim = 7
        ct = ObsTerm(func=cts_moe_task_context)  # dim = 1

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class Depth_ObsCfg(ObsGroup):
        """Depth image observation for CTS-MoE student encoder."""

        depth_image = ObsTerm(
            func=mdp.cts_moe_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_image"),
                "data_type": "distance_to_image_plane",
            },
        )  # shape: [B, 1, 58, 87]

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    @configclass
    class HeightScan_ObsCfg(ObsGroup):
        """Three-layer height scan observation for CTS-MoE teacher encoder."""

        height_scan = ObsTerm(
            func=mdp.cts_moe_height_scan,
            params={
                "sensor_cfgs": [
                    SceneEntityCfg("H_ground_scan"),
                ],
                "offset": 0.5,
            },
        )  # shape: [B, 1, num_rays, 1]

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    proprio: ProprioCfg = ProprioCfg()
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    privileged_obs: PrivilegedObsCfg = PrivilegedObsCfg()

    depth: Depth_ObsCfg = Depth_ObsCfg()
    height_scan: HeightScan_ObsCfg = HeightScan_ObsCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- ARM 
    # The name must have a prefix of "end_effector_".
    end_effector_position_tracking_exp_common = RewTerm(
        func=mdp.position_command_error_exp,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "command_name": "ee_pose",
                "std": 0.1},
    )

    end_effector_position_tracking_l2_common = RewTerm(
        func=mdp.position_command_error_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "command_name": "ee_pose"},
    )

    end_effector_position_tracking_fine_grained_common = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "std": 0.1,  
            "command_name": "ee_pose",
        },
    )

    end_effector_orientation_tracking_flat = RewTerm(
        func=mdp.orientation_command_error,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "command_name": "ee_pose"},
    )

    end_effector_action_rate_common = RewTerm(func=mdp.action_rate_l2_arm, weight=-0.005)

    end_effector_action_smoothness_common = RewTerm(func=mdp.arm_action_smoothness_penalty, weight=-0.02)

    end_effector_joint_vel_common = RewTerm(
        func=mdp.joint_vel_l2_arm,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    end_effector_lin_vel_z_l2_common = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.5)
    end_effector_ang_vel_xy_l2_common = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02) # -0.05
    end_effector_flat_orientation_l2_common = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # arm_contact = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-2.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link.*","end_effector"]), "threshold": 0.5},
    # )

    # base_contact = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.5},
    # )

    # more rewards
    # end_effector_xxx = xxx


    # -- LEG
    track_lin_vel_x_exp_box_avoidance = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_lin_vel_y_exp_box_avoidance = RewTerm(
        func=mdp.track_lin_vel_y_exp,
        weight=0.2,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_lin_vel_x_exp_under_table = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_lin_vel_y_exp_under_table = RewTerm(
        func=mdp.track_lin_vel_y_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_lin_vel_x_exp_stair_up = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_lin_vel_y_exp_stair_up = RewTerm(
        func=mdp.track_lin_vel_y_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_lin_vel_x_exp_flat = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_lin_vel_y_exp_flat = RewTerm(
        func=mdp.track_lin_vel_y_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )


    
    track_ang_vel_z_exp_common = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=1.5,
         params={ 
                 "command_name": "base_velocity", 
                 "std": math.sqrt(0.2)}
    )

    track_ori_exp_common = RewTerm(
        func=mdp.base_ori_tracking, 
        weight=0.5,
         params={ 
                 "command_name": "ee_pose", 
                 "std": 0.2}
    )

    track_base_height_exp_box_avoidance = RewTerm(
        func=mdp.base_height_tracking, 
        weight=0.5,
         params={ 
                 "desired_height": 0.3, 
                 "std": 0.02}
    )

    track_base_height_exp_under_table = RewTerm(
        func=mdp.base_height_tracking_in_table_region,
        weight=0.5,
        params={
            "desired_height": 0.22,
            "std": 0.02,
            "table_half_extents_xy": (1.0 + 0.5, 0.7),
        },
    )

    track_base_height_exp_stair_up = RewTerm(
        func=mdp.base_height_tracking, 
        weight=0.5,
         params={ 
                 "desired_height": 0.3, 
                 "std": 0.02}
    )

    forward_progress_stair_up = RewTerm(
        func=mdp.stair_up_forward_progress,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "first_step_object_name": "stair_step_0",
            "step_depth": 0.35,
            "num_steps": 7,
            "approach_margin": 0.4,
            "platform_margin": 0.7,
        },
    )

    base_height_progress_stair_up = RewTerm(
        func=mdp.stair_up_base_height_progress,
        weight=0.0,
        params={
            "desired_base_clearance": 0.3,
            "std": 0.06,
            "asset_cfg": SceneEntityCfg("robot"),
            "first_step_object_name": "stair_step_0",
            "step_depth": 0.35,
            "step_height": 0.05,
            "num_steps": 7,
            "approach_margin": 0.2,
        },
    )

    track_base_height_exp_flat = RewTerm(
        func=mdp.base_height_tracking, 
        weight=0.5,
         params={ 
                 "desired_height": 0.3, 
                 "std": 0.02}
    )


    lin_vel_z_l2_box_avoidance = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.5)
    lin_vel_z_l2_under_table = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    lin_vel_z_l2_stair_up = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    lin_vel_z_l2_flat = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)


    ang_vel_xy_l2_common = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02) # -0.05
    dof_torques_l2_common = RewTerm(func=mdp.joint_torques_l2_Go2, weight=-2.0e-5) # - 0.0002
    dof_acc_l2_common = RewTerm(func=mdp.joint_acc_l2_Go2, weight=-2.5e-7)
    action_rate_l2_common = RewTerm(func=mdp.action_rate_l2_Go2, weight=-0.01)

    feet_air_time_common = RewTerm(
        func=mdp.feet_air_time,
        weight= 0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    feet_slide_common = RewTerm(
        func=mdp.feet_slide,
        weight= -0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    F_feet_air_time_common = RewTerm(
        func=mdp.feet_air_time,
        weight= 0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    R_feet_air_time_common = RewTerm(
        func=mdp.feet_air_time,
        weight= 2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    feet_height_flat = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": 0.08,
            "command_name": "base_velocity",
        },
    )
    feet_height_under_table = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": 0.08,
            "command_name": "base_velocity",
        },
    )
    feet_height_box_avoidance = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": 0.08,
            "command_name": "base_velocity",
        },
    )

    feet_long_air_common = RewTerm(
        func=mdp.feet_long_air_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_air_time": 0.5,
        },
    )
    

    feet_height_body_stair_up = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": -0.15,
            "command_name": "base_velocity",
        },
    )
    

    # feet_height_common = RewTerm(
    #     func=mdp.feet_height,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "tanh_mult": 2.0,
    #         "target_height": 0.08,
    #         "command_name": "base_velocity",
    #     },
    # )
    
    foot_contact_common = RewTerm(
        func=mdp.standing_feet_contact_force,
        weight= 0.003,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "force_threshold": 10.0,
            "command_threshold": 0.1,
        },
    )

    joint_mirror_common = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    gait_reward_common = RewTerm(
        func=mdp.GaitReward,
        weight=0.1,
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
        params={
            "std": 0.1,
            "command_name": "base_velocity",
            "max_err": 0.5,
            "velocity_threshold": 0.1,
            "command_threshold": 0.1,            
            "synced_feet_pair_names": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },    
    )

    hip_deviation_common = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

    joint_deviation_common = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )

    F_joint_deviation_common = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["F.*_thigh_joint", "F.*_calf_joint"])},
    )


    R_joint_deviation_common = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["R.*_thigh_joint", "R.*_calf_joint"])},
    )


    action_smoothness_common = RewTerm(
        func=mdp.leg_action_smoothness_penalty,
        weight=-0.02,
    )
    
    flat_orientation_l2_box_avoidance = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    flat_orientation_l2_under_table = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    flat_orientation_l2_stair_up = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    flat_orientation_l2_flat = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # height_reward = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.31, "std":0.03})

    # height_exp_reward = RewTerm(func=mdp.base_height_exp, weight=-2.0, params={"target_height": 0.31, "std":0.03})

    probe_clearance_under_table = RewTerm(
        func=mdp.probe_links_below_height_exp_in_table_region,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["probe0", "probe1", "probe2"]),
            "max_height": 0.5,
            "std": 0.05,
            "table_half_extents_xy": (1.0 + 0.5, 0.7),
        },
    )

   
    thigh_contact_box_avoidance = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 0.5},
    )
    thigh_contact_under_table = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 0.5},
    )
    thigh_contact_stair_up = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 0.5},
    )
    thigh_contact_flat = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 0.5},
    )



    calf_contact_box_avoidance = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 0.5},
    )
    calf_contact_under_table = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 0.5},
    )
    calf_contact_stair_up = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 0.5},
    )
    calf_contact_flat = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 0.5},
    )



    base_contact_box_avoidance = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.5},
    )
    base_contact_under_table = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.5},
    )
    base_contact_stair_up = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.5},
    )
    base_contact_flat = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.5},
    )


    arm_contact_box_avoidance = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link[3-6]", "end_effector"]), "threshold": 0.5},
    )
    arm_contact_under_table = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link[3-6]", "end_effector"]), "threshold": 0.5},
    )
    arm_contact_stair_up = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link[3-6]", "end_effector"]), "threshold": 0.5},
    )
    arm_contact_flat = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link[3-6]", "end_effector"]), "threshold": 0.5},
    )

@configclass
class MultiTaskRewardCfg:
    """CTS-MoE task assignment settings and dispatcher-level reward terms."""

    # Dispatcher-level alive bonus added to every task after marked rewards are computed.
    alive_weight: float = 0.1

    # task assignment
    fixed_task_assignment: bool = True
    fixed_task_id: int | None = None
    task_sampling_weights: list[float] | None = None
    enable_box_avoidance: bool = True
    stair_step_height: float = 0.05
    stair_step_depth: float = 0.35
    stair_num_steps: int = 7


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.5},
    # )

    # thigh_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold":0.5},
    # )
    
    # calf_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 0.5},
    # )

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})

    # arm_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link[3-6]","end_effector"]), "threshold": 0.5},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # flat_ori_modify = CurrTerm(func=mdp.modify_reward_weight,
    #                            params={"term_name": "flat_orientation_l2",
    #                                    "num_steps": 3500,
    #                                    "weight": -0.75})
    # joint_deviation_modify = CurrTerm(func=mdp.modify_reward_weight,
    #                            params={"term_name": "joint_deviation",
    #                                    "num_steps": 3500,
    #                                    "weight": -0.03})
    # hip_deviation_modify = CurrTerm(func=mdp.modify_reward_weight,
    #                            params={"term_name": "hip_deviation",
    #                                    "num_steps": 3500,
    #                                    "weight": -0.2})

    # track_ori_exp_modify = CurrTerm(func=mdp.modify_reward_weight,
    #                            params={"term_name": "track_ori_exp",
    #                                    "num_steps": 2500,
    #                                    "weight": 1.25 })

    # flat_orientation_modify = CurrTerm(func=mdp.modify_reward_weight,
    #                            params={"term_name": "flat_orientation_l2",
    #                                    "num_steps": 2500,
    #                                    "weight": -0.5})
    

##
# Environment configuration
##

@configclass
class LocomotionVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=10.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    multi_task_rewards: MultiTaskRewardCfg = MultiTaskRewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.H_ground_scan is not None:
            self.scene.H_ground_scan.update_period = self.decimation * self.sim.dt
        if self.scene.H_lateral_scan is not None:
            self.scene.H_lateral_scan.update_period = self.decimation * self.sim.dt
        if self.scene.H_overhead_scan is not None:
            self.scene.H_overhead_scan.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
