from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg_moe_ortho import Go2PiperCTSMoEAlgorithmCfg, Go2PiperCTSMoEPolicyCfg


@configclass
class Go2PiperDualRouterCTSMoEPolicyCfg(Go2PiperCTSMoEPolicyCfg):
    """Network config for history-basis CTS-MoE with separate leg/arm routers."""

    class_name: str = "StructureAwareDualRouterCTSMoEPolicy"

    # Split action heads.  The final action is concat([leg_action, arm_action]).
    leg_action_dim: int = 12
    arm_action_dim: int = 6


@configclass
class Go2PiperDualRouterCTSMoERunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for dual-router history-basis orthogonal CTS-MoE training."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "go2piper_cts_moe_ortho_dual_router_history_basis"
    empirical_normalization = False
    load_checkpoint: str = "CTSMoEDualRouter_.*.pt"

    policy = Go2PiperDualRouterCTSMoEPolicyCfg(
        proprio_dim=72,
        privileged_dim=104,
        action_dim=18,
        latent_dim=32,
        num_tasks=5,
        height_channels=1,
        teacher_privileged_hidden_dims=[512, 256],
        teacher_privileged_feature_dim=32,
        teacher_height_hidden_dims=[512, 256],
        teacher_height_feature_dim=128,
        actor_type="orthogonal_cts_moe",
        orthogonal_mode="gram_schmidt",
        gate_activation="tanh",
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
        critic_latent_source="teacher",
        leg_action_dim=12,
        arm_action_dim=6,
        init_log_std=0.0,
        learnable_log_std=True,
        activation="elu",
    )

    algorithm = Go2PiperCTSMoEAlgorithmCfg(
        training_mode="student_policy",
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
        router_entropy_coef=0.0,
        router_balance_coef=0.0,
        router_logit_l2_coef=1e-4,
        lambda_orth=1e-3,
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
