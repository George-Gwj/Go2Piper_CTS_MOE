from dataclasses import MISSING

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class Go2PiperCTSMoEPolicyCfg:
    """Network config for a single full-body StructureAwareCTSMoEPolicy."""

    class_name: str = "StructureAwareCTSMoEPolicy"

    # Full-body action/proprio dimensions. proprio_dim is expected to include commands.
    proprio_dim: int = 66
    privileged_dim: int = 98
    action_dim: int = 18

    # Shared latent/task dimensions.
    latent_dim: int = 32
    num_tasks: int = 4
    activation: str = "elu"

    # Teacher encoder: MLP(e_t), MLP(h_t), optional c_t, then Linear + LayerNorm.
    height_channels: int = 3
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

    # Dense MoE actor. Router uses z only; each expert outputs full-body action.
    num_experts: int = 4
    expert_hidden_dims: list[int] = [256, 128]
    router_hidden_dims: list[int] = [128, 64]
    expert_names: list[str] = ["lateral_avoidance", "under_table", "stair_up", "stair_down"]

    # Sparse multi-critic.
    critic_hidden_dims: list[int] = [256, 128]
    critic_shared_trunk: bool = False
    critic_trunk_hidden_dims: list[int] | None = None
    critic_head_hidden_dims: list[int] = [64]

    # Gaussian policy std.
    init_log_std: float = 0.0
    learnable_log_std: bool = True


@configclass
class Go2PiperCTSMoEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Algorithm config consumed by CTSMoEPPO."""

    class_name: str = "CTSMoEPPO"

    # PPO.
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
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
    """Runner config for a single CTS-MoE full-body policy."""

    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 100
    experiment_name = "go2piper_cts_moe"
    empirical_normalization = False

    policy = Go2PiperCTSMoEPolicyCfg(
        proprio_dim=66,
        privileged_dim=98,
        action_dim=18,
        latent_dim=32,
        num_tasks=4,
        height_channels=3,
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
        num_experts=4,
        expert_hidden_dims=[256, 128],
        router_hidden_dims=[128, 64],
        critic_hidden_dims=[256, 128],
        critic_shared_trunk=False,
        critic_head_hidden_dims=[64],
        init_log_std=0.0,
        learnable_log_std=True,
        activation="elu",
    )

    algorithm = Go2PiperCTSMoEAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        eps=1e-5,
        student_learning_rate=1e-4,
        distillation_loss_coef=1.0,
        student_rollout_ratio=0.15,
        router_entropy_coef=5e-4,
        router_balance_coef=2e-3,
        router_logit_l2_coef=1e-5,
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
