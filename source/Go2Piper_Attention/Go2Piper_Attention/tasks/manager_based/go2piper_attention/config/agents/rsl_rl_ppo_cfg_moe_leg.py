from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_moe_ortho import (
    Go2PiperCTSMoEAlgorithmCfg,
    Go2PiperCTSMoEPolicyCfg,
    Go2PiperCTSMoERunnerCfg as Go2PiperCTSMoEOrthoRunnerCfg,
    Go2PiperCTSMoETeacherAlgorithmCfg,
    Go2PiperCTSMoETeacherPolicyCfg,
    Go2PiperCTSMoETeacherRunnerCfg as Go2PiperCTSMoEOrthoTeacherRunnerCfg,
)


@configclass
class Go2PiperLegCTSMoEPolicyCfg(Go2PiperCTSMoEPolicyCfg):
    """Network config for the leg-only action-level CTS-MoE policy."""


@configclass
class Go2PiperLegCTSMoETeacherPolicyCfg(Go2PiperCTSMoETeacherPolicyCfg):
    """Network config for teacher-only leg-only action-level CTS-MoE training."""


@configclass
class Go2PiperLegCTSMoEAlgorithmCfg(Go2PiperCTSMoEAlgorithmCfg):
    """Algorithm config for leg-only CTS-MoE with anti-collapse router regularization."""

    router_entropy_coef: float = 5.0e-4
    router_balance_coef: float = 2.0e-3
    router_logit_l2_coef: float = 1.0e-4
    lambda_orth: float = 0.0


@configclass
class Go2PiperLegCTSMoETeacherAlgorithmCfg(Go2PiperCTSMoETeacherAlgorithmCfg):
    """Teacher-only leg CTS-MoE config with anti-collapse router regularization."""

    router_entropy_coef: float = 5.0e-4
    router_balance_coef: float = 2.0e-3
    router_logit_l2_coef: float = 1.0e-4
    lambda_orth: float = 0.0


@configclass
class Go2PiperLegCTSMoERunnerCfg(Go2PiperCTSMoEOrthoRunnerCfg):
    """Runner config for leg-only action-level CTS-MoE training."""

    experiment_name = "go2piper_leg_cts_moe"
    load_checkpoint: str = "LegCTSMoE_.*.pt"

    policy = Go2PiperLegCTSMoEPolicyCfg(
        proprio_dim=46,
        privileged_dim=65,
        action_dim=12,
        latent_dim=32,
        num_tasks=5,
        height_channels=1,
        teacher_privileged_hidden_dims=[512, 256],
        teacher_privileged_feature_dim=32,
        teacher_height_hidden_dims=[512, 256],
        teacher_height_feature_dim=128,
        student_perception_type="depth",
        student_perception_channels=1,
        student_proprio_hidden_dims=[512, 256],
        student_proprio_feature_dim=32,
        student_depth_filters=[16, 32, 64],
        student_depth_feature_dim=128,
        student_gru_hidden_dim=256,
        student_gru_num_layers=1,
        actor_type="cts_moe",
        orthogonal_mode="none",
        gate_activation="softmax",
        num_experts=4,
        expert_names=[
            "expert_0",
            "expert_1",
            "expert_2",
            "expert_3",
        ],
        expert_feature_dim=128,
        expert_hidden_dims=[256, 128],
        router_hidden_dims=[128, 64],
        action_head_hidden_dims=[256, 128],
        use_expert_layernorm=True,
        use_moe_output_layernorm=True,
        gram_schmidt_eps=1e-6,
        log_expert_metrics=True,
        critic_hidden_dims=[256, 128],
        critic_shared_trunk=False,
        critic_head_hidden_dims=[64],
        init_log_std=0.0,
        learnable_log_std=True,
        activation="elu",
    )

    algorithm = Go2PiperLegCTSMoEAlgorithmCfg()


@configclass
class Go2PiperLegCTSMoETeacherRunnerCfg(Go2PiperCTSMoEOrthoTeacherRunnerCfg):
    """Runner config for teacher-only leg-only action-level CTS-MoE training."""

    experiment_name = "go2piper_leg_cts_moe_teacher"
    load_checkpoint: str = "LegCTSMoETeacher_.*.pt"

    policy = Go2PiperLegCTSMoETeacherPolicyCfg(
        proprio_dim=46,
        privileged_dim=65,
        action_dim=12,
        latent_dim=32,
        num_tasks=5,
        height_channels=1,
        teacher_privileged_hidden_dims=[512, 256],
        teacher_privileged_feature_dim=32,
        teacher_height_hidden_dims=[512, 256],
        teacher_height_feature_dim=128,
        student_perception_type="depth",
        student_perception_channels=1,
        student_proprio_hidden_dims=[512, 256],
        student_proprio_feature_dim=32,
        student_depth_filters=[16, 32, 64],
        student_depth_feature_dim=128,
        student_gru_hidden_dim=256,
        student_gru_num_layers=1,
        actor_type="cts_moe",
        orthogonal_mode="none",
        gate_activation="softmax",
        num_experts=4,
        expert_names=[
            "expert_0",
            "expert_1",
            "expert_2",
            "expert_3",
        ],
        expert_feature_dim=128,
        expert_hidden_dims=[256, 128],
        router_hidden_dims=[128, 64],
        action_head_hidden_dims=[256, 128],
        use_expert_layernorm=True,
        use_moe_output_layernorm=True,
        gram_schmidt_eps=1e-6,
        log_expert_metrics=True,
        critic_hidden_dims=[256, 128],
        critic_shared_trunk=False,
        critic_head_hidden_dims=[64],
        init_log_std=0.0,
        learnable_log_std=True,
        activation="elu",
    )

    algorithm = Go2PiperLegCTSMoETeacherAlgorithmCfg()


Go2PiperCTSMoERunnerCfg = Go2PiperLegCTSMoERunnerCfg
Go2PiperCTSMoETeacherRunnerCfg = Go2PiperLegCTSMoETeacherRunnerCfg
