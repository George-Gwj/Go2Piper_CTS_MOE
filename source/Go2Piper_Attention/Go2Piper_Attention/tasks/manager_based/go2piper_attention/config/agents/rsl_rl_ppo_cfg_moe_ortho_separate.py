from dataclasses import MISSING

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class Go2PiperCTSMoEPolicyCfg:
    """Network config for decoupled leg CTS-MoE and arm MLP policy."""

    class_name: str = "HybridLegArmCTSMoEPolicy"

    # Separate wrapper concatenates leg_proprio(58) and arm_proprio(31).
    proprio_dim: int = 89
    privileged_dim: int = 103
    leg_action_dim: int = 12
    arm_action_dim: int = 6

    # Shared latent/task dimensions.
    latent_dim: int = 32
    num_tasks: int = 5
    activation: str = "elu"

    # Teacher encoder: MLP(e_t), MLP(h_t), optional c_t, then Linear + LayerNorm.
    height_channels: int = 1
    teacher_context_dim: int = 0
    teacher_height_flat_dim: int | None = None
    semantic_decoupled_teacher: bool = False
    teacher_privileged_hidden_dims: list[int] = [512, 256]
    teacher_privileged_feature_dim: int = 32
    teacher_height_hidden_dims: list[int] = [512, 256]
    teacher_height_feature_dim: int = 128

    # Student encoder: MLP(o^p_{t-H:t}), depth CNN, GRU, then Linear + LayerNorm.
    student_perception_type: str = "depth"
    student_perception_dim: int | None = None
    student_perception_channels: int = 1
    student_proprio_hidden_dims: list[int] = [512, 256]
    student_proprio_feature_dim: int = 32
    student_depth_filters: list[int] = [16, 32, 64]
    student_depth_feature_dim: int = 128
    student_gru_hidden_dim: int = 256
    student_gru_num_layers: int = 1

    # Leg CTS-MoE actor.
    orthogonal_mode: str = "gram_schmidt"
    gate_activation: str = "tanh"

    # Dense MoE actor. Router uses z only. Orthogonal actor experts output features.
    num_experts: int = 3
    expert_feature_dim: int = 128
    expert_hidden_dims: list[int] = [256, 128]
    router_hidden_dims: list[int] = [128, 64]
    action_head_hidden_dims: list[int] = [256, 128]
    expert_names: list[str] = ["expert_0", "expert_1", "expert_2"]
    use_expert_layernorm: bool = True
    use_moe_output_layernorm: bool = True
    gram_schmidt_eps: float = 1e-6
    log_expert_metrics: bool = True

    # Leg sparse multi-critic.
    leg_critic_hidden_dims: list[int] = [256, 128]
    leg_critic_shared_trunk: bool = False
    leg_critic_trunk_hidden_dims: list[int] | None = None
    leg_critic_head_hidden_dims: list[int] = [64]

    # Arm MLP branch. By default it observes full proprio; indices can narrow this later.
    arm_actor_hidden_dims: list[int] = [256, 128]
    arm_critic_hidden_dims: list[int] = [256, 128]
    leg_actor_proprio_indices: list[int] | None = list(range(58))
    leg_critic_proprio_indices: list[int] | None = None
    arm_actor_obs_indices: list[int] | None = list(range(58, 89))
    arm_critic_obs_indices: list[int] | None = None
    arm_actor_use_latent: bool = True
    arm_critic_use_latent: bool = True
    arm_detach_latent: bool = True

    # Gaussian policy std.
    init_leg_log_std: float = 0.0
    init_arm_log_std: float = 0.0
    learnable_log_std: bool = True


@configclass
class Go2PiperCTSMoETeacherPolicyCfg(Go2PiperCTSMoEPolicyCfg):
    """Network config for teacher-only CTS-MoE training (same architecture, no student rollout)."""


@configclass
class Go2PiperCTSMoEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Algorithm config consumed by HybridLegArmPPO."""

    class_name: str = "HybridLegArmPPO"

    # teacher: privileged teacher encoder only; mixed: teacher PPO + student distillation.
    training_mode: str = "mixed"

    # PPO.
    leg_value_loss_coef: float = 1.0
    arm_value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    leg_entropy_coef: float = 0.005
    arm_entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3.0e-4
    leg_learning_rate: float | None = None
    arm_learning_rate: float | None = 3.0e-4
    student_learning_rate: float | None = 1.0e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    eps: float = 1e-5

    # Student distillation.
    distillation_loss_coef: float = 1.0
    student_rollout_ratio: float = 0.15

    # Router auxiliary losses.
    router_entropy_coef: float = 0.0
    router_balance_coef: float = 0.0
    router_logit_l2_coef: float = 0.0
    lambda_orth: float = 0.0
    orth_loss_on: str = "raw"

    # Advantage normalization.
    per_task_leg_advantage_normalization: bool = True
    normalize_arm_advantage: bool = True


@configclass
class Go2PiperCTSMoERunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for a single CTS-MoE full-body policy."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "go2piper_cts_moe_ortho_separate"
    empirical_normalization = False
    load_checkpoint: str = "CTSMoEOrthoSeparate_.*.pt"

    policy = Go2PiperCTSMoEPolicyCfg(
        proprio_dim=89,
        privileged_dim=103,
        leg_action_dim=12,
        arm_action_dim=6,
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
        orthogonal_mode="gram_schmidt",
        gate_activation="tanh",
        num_experts=6,
        expert_names=[
            "expert_0",
            "expert_1",
            "expert_2",
            "expert_3",
            "expert_4",
            "expert_5",
        ],
        expert_feature_dim=128,
        expert_hidden_dims=[256, 128],
        router_hidden_dims=[128, 64],
        action_head_hidden_dims=[256, 128],
        use_expert_layernorm=True,
        use_moe_output_layernorm=True,
        gram_schmidt_eps=1e-6,
        log_expert_metrics=True,
        leg_critic_hidden_dims=[256, 128],
        leg_critic_shared_trunk=False,
        leg_critic_head_hidden_dims=[64],
        arm_actor_hidden_dims=[256, 128],
        arm_critic_hidden_dims=[256, 128],
        arm_actor_use_latent=True,
        arm_critic_use_latent=True,
        arm_detach_latent=True,
        init_leg_log_std=0.0,
        init_arm_log_std=0.0,
        learnable_log_std=True,
        activation="elu",
    )

    algorithm = Go2PiperCTSMoEAlgorithmCfg(
        training_mode="mixed",
        leg_value_loss_coef=1.0,
        arm_value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        leg_entropy_coef=0.005,
        arm_entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        leg_learning_rate=None,
        arm_learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        eps=1e-5,
        student_learning_rate=1e-4,
        distillation_loss_coef=1.0,
        student_rollout_ratio=0.2,
        router_entropy_coef=0.0,
        router_balance_coef=0.0,
        router_logit_l2_coef=0.0,
        lambda_orth=1e-3,
        orth_loss_on="raw",
        per_task_leg_advantage_normalization=True,
        normalize_arm_advantage=True,
    )


@configclass
class Go2PiperCTSMoETeacherAlgorithmCfg(Go2PiperCTSMoEAlgorithmCfg):
    """Algorithm config for teacher-only CTS-MoE training."""

    training_mode: str = "teacher"
    student_rollout_ratio: float = 0.0
    distillation_loss_coef: float = 0.0
    num_mini_batches: int = 4


@configclass
class Go2PiperCTSMoETeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for teacher-only CTS-MoE training."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "go2piper_cts_moe_ortho_separate_teacher"
    empirical_normalization = False
    load_checkpoint: str = "CTSMoEOrthoSeparateTeacher_.*.pt"

    policy = Go2PiperCTSMoETeacherPolicyCfg(
        proprio_dim=89,
        privileged_dim=103,
        leg_action_dim=12,
        arm_action_dim=6,
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
        orthogonal_mode="gram_schmidt",
        gate_activation="tanh",
        num_experts=6,
        expert_feature_dim=128,
        expert_names=[
            "expert_0",
            "expert_1",
            "expert_2",
            "expert_3",
            "expert_4",
            "expert_5",
        ],
        expert_hidden_dims=[256, 128],
        router_hidden_dims=[128, 64],
        action_head_hidden_dims=[256, 128],
        use_expert_layernorm=True,
        use_moe_output_layernorm=True,
        gram_schmidt_eps=1e-6,
        log_expert_metrics=True,
        leg_critic_hidden_dims=[256, 128],
        leg_critic_shared_trunk=False,
        leg_critic_head_hidden_dims=[64],
        arm_actor_hidden_dims=[256, 128],
        arm_critic_hidden_dims=[256, 128],
        arm_actor_use_latent=True,
        arm_critic_use_latent=True,
        arm_detach_latent=True,
        init_leg_log_std=0.0,
        init_arm_log_std=0.0,
        learnable_log_std=True,
        activation="elu",
    )

    algorithm = Go2PiperCTSMoETeacherAlgorithmCfg(
        training_mode="teacher",
        leg_value_loss_coef=1.0,
        arm_value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        leg_entropy_coef=0.005,
        arm_entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        leg_learning_rate=None,
        arm_learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        eps=1e-5,
        student_learning_rate=1e-4,
        distillation_loss_coef=0.0,
        student_rollout_ratio=0.0,
        router_entropy_coef=0.0, # TODO 5e-4
        router_balance_coef=0.0, # TODO 2e-3
        router_logit_l2_coef=0.0,
        lambda_orth=1e-3,
        orth_loss_on="raw",
        per_task_leg_advantage_normalization=True,
        normalize_arm_advantage=True,
    )
