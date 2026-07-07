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

