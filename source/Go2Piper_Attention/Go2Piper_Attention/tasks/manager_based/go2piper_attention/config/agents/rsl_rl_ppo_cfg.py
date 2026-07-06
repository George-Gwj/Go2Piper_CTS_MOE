from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass

@configclass
class Go2PiperRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    # leg_control_head_hidden_dims : list[int] = MISSING
    # """The hidden dimensions of the leg control head network."""

    # arm_control_head_hidden_dims : list[int] = MISSING
    # """The hidden dimensions of the arm control head network."""

    actor_mlp_hidden_dims: list[int] = MISSING

    actor_attn_hidden_dims: list[int] = MISSING

    critic_leg_control_head_hidden_dims : list[int] = None
    """The hidden dimensions of the critic leg control head network."""

    critic_arm_control_head_hidden_dims : list[int] = None
    """The hidden dimensions of the critic arm control head network."""

    num_leg_actions : int = MISSING
    """The number of leg actions."""

    num_arm_actions : int = MISSING
    """The number of arm actions."""

    init_noise_std : float = MISSING
    """The init_noise_std of  actions."""

    min_std: float = MISSING,
    """The min_std of  actions."""

    attn_feature_output: int = None

    rma_actor_hidden_dims: list[int] = None

    rma_control_head_hidden_dims: list[int] = None

    priv_encoder_dims: list[int] = MISSING

    control_head_hidden_dims : list[int] = None

    arm_control_head_hidden_dims : list[int] = None

    critic_control_head_hidden_dims : list[int] = None

    critic_attn_hidden_dims = [32], 
    critic_attn_output = [32],
    attn_output = 32,

@configclass
class Go2PiperRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    dagger_update_freq : int = None
    """The frequency of dagger update."""

    priv_reg_coef_schedual: list = None
    """The schedule of the privileged regularization coefficient."""

    mixing_schedule : list = None
    """The schedule of the mixing coefficient."""

    eps : float = None
    """The epsilon value for numerical stability."""


@configclass
class Go2PiperRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    leg_policy: Go2PiperRslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    leg_algorithm: Go2PiperRslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
    
    arm_policy: Go2PiperRslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    arm_algorithm: Go2PiperRslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
    
    leg_load_checkpoint :str = "Leg_.*.pt"

    arm_load_checkpoint :str = "Arm_.*.pt"

    options: str = MISSING
    """leg / arm / all"""

@configclass
class Go2PiperFlatPPORunnerCfg(Go2PiperRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 100
    experiment_name = "go2piper_attn"
    empirical_normalization = False

    leg_policy = Go2PiperRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_mlp_hidden_dims = [256, 128],

        actor_attn_hidden_dims = [32],
        rma_actor_hidden_dims = [256],
        rma_control_head_hidden_dims = [128],
        attn_feature_output = 16,
        priv_encoder_dims = [128,32],
        critic_hidden_dims = [256,128],
        critic_attn_hidden_dims = [128], 
        critic_attn_output = 64,
        attn_output = 32,
        # TODO: 
        num_leg_actions = 12,
        activation="elu",
        min_std = 0.05,
    )

    leg_algorithm = Go2PiperRslRlPpoAlgorithmCfg(
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
        dagger_update_freq = 20,
        priv_reg_coef_schedual = [0, 0.1, 1000, 3000],
        mixing_schedule=[1.0, 0, 4000] ,
        eps = 1e-5,
    )
    

    arm_policy = Go2PiperRslRlPpoActorCriticCfg(
        # class_name="ActorCriticArm",
        init_noise_std=1.0,
        actor_mlp_hidden_dims = [256, 128],

        actor_attn_hidden_dims = [32],
        rma_actor_hidden_dims = [256],
        rma_control_head_hidden_dims = [128],
        attn_feature_output = 16,
        priv_encoder_dims = [128,32],
        critic_hidden_dims = [256,128],
        critic_attn_hidden_dims = [32], 
        critic_attn_output = 32,
        attn_output = 32,

        num_arm_actions = 6,
        activation="elu",
        min_std = 0.05,

    )
    
    arm_algorithm = Go2PiperRslRlPpoAlgorithmCfg( # TODO: Go2PiperRslRlPpoAlgorithmCfg
        # class_name="PPOARM",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        dagger_update_freq = 20,
        priv_reg_coef_schedual = [0, 0.1, 500, 2000],
        mixing_schedule=[1.0, 0, 4000] ,
        eps = 1e-5,
    )



@configclass
class NoPriv_Go2PiperFlatPPORunnerCfg(Go2PiperRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 100
    experiment_name = "go2piper_attn_no_priv"
    empirical_normalization = False

    leg_policy = Go2PiperRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_mlp_hidden_dims = [256, 128],

        actor_attn_hidden_dims = [64],
        rma_actor_hidden_dims = [256],
        rma_control_head_hidden_dims = [128],
        attn_feature_output = 16,
        critic_hidden_dims = [256,128],
        critic_attn_hidden_dims = [128], 
        critic_attn_output = 64,
        attn_output = 32,
        num_leg_actions = 12,
        activation="elu",
        min_std = 0.2,
    )

    leg_algorithm = Go2PiperRslRlPpoAlgorithmCfg(
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
        eps = 1e-5,
    )
     
    arm_policy = Go2PiperRslRlPpoActorCriticCfg(
        # class_name="ActorCriticArm",
        init_noise_std=1.0,
        actor_mlp_hidden_dims = [256, 128],

        actor_attn_hidden_dims = [64],
        rma_actor_hidden_dims = [256],
        rma_control_head_hidden_dims = [128],
        attn_feature_output = 16,
        critic_hidden_dims = [256,128],
        critic_attn_hidden_dims = [32], 
        critic_attn_output = 32,
        attn_output = 32,

        num_arm_actions = 6,
        activation="elu",
        min_std = 0.2,

    )
    
    arm_algorithm = Go2PiperRslRlPpoAlgorithmCfg( # TODO: Go2PiperRslRlPpoAlgorithmCfg
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        eps = 1e-5,
    )

