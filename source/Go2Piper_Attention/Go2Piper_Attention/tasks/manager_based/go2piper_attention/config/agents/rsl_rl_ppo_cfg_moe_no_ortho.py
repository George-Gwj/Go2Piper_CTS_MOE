from dataclasses import MISSING

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class Go2PiperCTSMoEPolicyCfg:
    """Network config for a single full-body StructureAwareCTSMoEPolicy."""

    class_name: str = "StructureAwareCTSMoEPolicy"

    # Full-body action/proprio dimensions. proprio_dim is expected to include commands.
    proprio_dim: int = 71
    privileged_dim: int = 103
    action_dim: int = 18

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

    # Keep the feature-level MoE actor, but disable the Gram-Schmidt feature projection.
    actor_type: str = "orthogonal_cts_moe"
    orthogonal_mode: str = "none"
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

    # Sparse multi-critic.
    critic_hidden_dims: list[int] = [256, 128]
    critic_shared_trunk: bool = False
    critic_trunk_hidden_dims: list[int] | None = None
    critic_head_hidden_dims: list[int] = [64]

    # Gaussian policy std.
    init_log_std: float = 0.0
    learnable_log_std: bool = True


@configclass
class Go2PiperCTSMoETeacherPolicyCfg(Go2PiperCTSMoEPolicyCfg):
    """Network config for teacher-only CTS-MoE training (same architecture, no student rollout)."""


@configclass
class Go2PiperCTSMoEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Algorithm config consumed by CTSMoEPPO."""

    class_name: str = "CTSMoEPPO"

    # teacher: privileged teacher encoder only; mixed: teacher PPO + student distillation.
    training_mode: str = "mixed"

    # PPO.
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3.0e-4
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

    # Per-task advantage normalization.
    per_task_advantage_normalization: bool = True

    # Per-task POPArt return normalization.
    use_popart: bool = False
    popart_beta: float = 0.99999
    popart_eps: float = 1e-5
    popart_min_std: float = 1e-2
    popart_use_output_rescale: bool = True
    popart_value_loss: str = "huber"
    popart_huber_delta: float = 1.0
    value_loss_per_task_average: bool = True


@configclass
class Go2PiperCTSMoERunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for a no-orthogonalization CTS-MoE full-body policy."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "go2piper_cts_moe_no_ortho"
    empirical_normalization = False
    load_checkpoint: str = "CTSMoENoOrtho_.*.pt"

    policy = Go2PiperCTSMoEPolicyCfg(
        proprio_dim=71,
        privileged_dim=103,
        action_dim=18,
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
        actor_type="orthogonal_cts_moe",
        orthogonal_mode="none",
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
        critic_hidden_dims=[256, 128],
        critic_shared_trunk=False,
        critic_head_hidden_dims=[64],
        init_log_std=0.0,
        learnable_log_std=True,
        activation="elu",
    )

    algorithm = Go2PiperCTSMoEAlgorithmCfg(
        training_mode="mixed",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
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
        lambda_orth=0.0,
        orth_loss_on="raw",
        per_task_advantage_normalization=True,
        use_popart=True,
        popart_beta=0.99,
        popart_eps=1e-5,
        popart_min_std=1e-2,
        popart_use_output_rescale=True,
        popart_value_loss="huber",
        popart_huber_delta=1.0,
        value_loss_per_task_average=True,
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
    """Runner config for teacher-only no-orthogonalization CTS-MoE training."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "go2piper_cts_moe_no_ortho_teacher"
    empirical_normalization = False
    load_checkpoint: str = "CTSMoENoOrthoTeacher_.*.pt"

    policy = Go2PiperCTSMoETeacherPolicyCfg(
        proprio_dim=71,
        privileged_dim=103,
        action_dim=18,
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
        actor_type="orthogonal_cts_moe",
        orthogonal_mode="none",
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
        critic_hidden_dims=[256, 128],
        critic_shared_trunk=False,
        critic_head_hidden_dims=[64],
        init_log_std=0.0,
        learnable_log_std=True,
        activation="elu",
    )

    algorithm = Go2PiperCTSMoETeacherAlgorithmCfg(
        training_mode="teacher",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
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
        lambda_orth=0.0,
        orth_loss_on="raw",
        per_task_advantage_normalization=True,
        use_popart=True,
        popart_beta=0.99,
        popart_eps=1e-5,
        popart_min_std=1e-2,
        popart_use_output_rescale=True,
        popart_value_loss="huber",
        popart_huber_delta=1.0,
        value_loss_per_task_average=True,
    )
