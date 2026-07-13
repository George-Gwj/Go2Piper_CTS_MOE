# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacAttention/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Go2Piper-Attention",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2PiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_env_cfg:Go2PiperMoEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe:Go2PiperCTSMoERunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Teacher",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_env_cfg:Go2PiperMoEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe:Go2PiperCTSMoETeacherRunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_env_cfg:Go2PiperMoEEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe:Go2PiperCTSMoERunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Teacher-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_env_cfg:Go2PiperMoEEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe:Go2PiperCTSMoETeacherRunnerCfg",
    },
)


gym.register(
    id="Go2Piper-Attention-CTS-MoE-Ortho",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_ortho_env_cfg:Go2PiperMoEOrthoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_ortho:Go2PiperCTSMoERunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Ortho-Teacher",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_ortho_env_cfg:Go2PiperMoEOrthoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_ortho:Go2PiperCTSMoETeacherRunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Ortho-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_ortho_env_cfg:Go2PiperMoEOrthoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_ortho:Go2PiperCTSMoERunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Ortho-Teacher-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_ortho_env_cfg:Go2PiperMoEOrthoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_ortho:Go2PiperCTSMoETeacherRunnerCfg",
    },
)


gym.register(
    id="Go2Piper-Attention-CTS-MoE-Ortho-CNN",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_ortho_env_CNN_cfg:Go2PiperMoEOrthoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_ortho_CNN:Go2PiperCTSMoERunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Ortho-CNN-Teacher",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_ortho_env_CNN_cfg:Go2PiperMoEOrthoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_ortho_CNN:Go2PiperCTSMoETeacherRunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Ortho-CNN-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_ortho_env_CNN_cfg:Go2PiperMoEOrthoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_ortho_CNN:Go2PiperCTSMoERunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Ortho-CNN-Teacher-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_ortho_env_CNN_cfg:Go2PiperMoEOrthoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_ortho_CNN:Go2PiperCTSMoETeacherRunnerCfg",
    },
)


gym.register(
    id="Go2Piper-Attention-CTS-MoE-Leg-Ortho",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_leg_ortho_env_cfg:Go2PiperMoEOrthoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_leg_ortho:Go2PiperLegCTSMoERunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Leg-Ortho-Teacher",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_leg_ortho_env_cfg:Go2PiperMoEOrthoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_leg_ortho:Go2PiperLegCTSMoETeacherRunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Leg-Ortho-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_leg_ortho_env_cfg:Go2PiperMoEOrthoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_leg_ortho:Go2PiperLegCTSMoERunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-CTS-MoE-Leg-Ortho-Teacher-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moe_leg_ortho_env_cfg:Go2PiperMoEOrthoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_moe_leg_ortho:Go2PiperLegCTSMoETeacherRunnerCfg",
    },
)


gym.register(
    id="Go2Piper-Attention-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2PiperFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-No-Priv",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2PiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NoPriv_Go2PiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2Piper-Attention-No-Priv-Play",
    entry_point="Go2Piper_Attention.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2PiperFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NoPriv_Go2PiperFlatPPORunnerCfg",
    },
)
