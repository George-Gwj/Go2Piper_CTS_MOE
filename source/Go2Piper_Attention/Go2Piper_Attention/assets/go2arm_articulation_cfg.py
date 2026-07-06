import os
current_dir = os.path.dirname(os.path.abspath(__file__))
GO2PIPER_USD = os.path.join(current_dir, "go2_piper_camera.usd")

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##


GO2PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2PIPER_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     collision_enabled=True,
        #     contact_offset=0.02,
        #     rest_offset=0.005 ,
        # ),
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={
            # leg
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            # arm
            # "joint1":0.0,
            # "joint2":1.1,# 0.8
            # "joint3":-0.8,# -1.0
            # "joint4":-0.0,
            # "joint5":-0.7,
            # "joint6":0.0,
            "joint1":0.0,
            "joint2":0.0,# 0.8
            "joint3":-0.0,# -1.0
            "joint4":-0.0,
            "joint5":-0.0,
            "joint6":0.0,
        },
        joint_vel={".*": 0.0},
    ),
    
    soft_joint_pos_limit_factor=0.9,
    actuators={

        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=50.0,
            damping=1.2,
            armature=0.01,
        ),

        "piper_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint1","joint2","joint3", "joint5"],
            stiffness=80.0,
            damping=3.0,
            armature=0.01,
        ),

        "piper_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint4","joint6"],
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),

        # "base_legs": DCMotorCfg(
        #     joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        #     effort_limit=45,
        #     saturation_effort=45,
        #     velocity_limit=30.0,
        #     stiffness=50.0,
        #     damping=1.2,
        #     armature=0.01
        # ),


        # "base_legs": DelayedPDActuatorCfg(
        #     joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        #     stiffness=50.0,
        #     damping=1.0,
        #     armature=0.01,
        #     min_delay=0,
        #     max_delay=3
        # ),

        # "base_legs": IdealPDActuatorCfg(
        #     joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        #     stiffness=50.0,
        #     damping=1.2,
        #     armature=0.01,
        # ),


        # "piper_shoulder": DCMotorCfg(
        #     joint_names_expr=["joint1","joint2","joint3","joint5"],
        #     effort_limit=40.0,
        #     saturation_effort = 40.0,
        #     velocity_limit=20,
        #     stiffness=50.0,
        #     damping=1.0,
        #     armature=0.01,
        # ),

        # "piper_forearm": DCMotorCfg(
        #     joint_names_expr=["joint4","joint6"],
        #     effort_limit=40.0,
        #     saturation_effort=40.0,
        #     velocity_limit=20,
        #     stiffness=5.0,
        #     damping=0.2,
        #     armature=0.01,
        # ),

    },
)

