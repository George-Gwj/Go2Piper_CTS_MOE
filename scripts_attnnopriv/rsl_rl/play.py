# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import argparse
import sys
import math
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
from local_rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# from isaaclab_rl.rsl_rl import export_policy_as_jit
from exporter import export_policy_as_jit
from local_rsl_rl.wrappers import RslRlVecEnvWrapper

from Go2Piper_Attention.tasks.manager_based.go2piper_attention.config.agents import Go2PiperRslRlOnPolicyRunnerCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Go2Piper_Attention.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: Go2PiperRslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint: # TODO: not use now 
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint) # TODO not use now
    else:
        if args_cli.options == "leg":
            leg_resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.leg_load_checkpoint)
            log_dir = os.path.dirname(leg_resume_path)
            print(f"[INFO]: Loading leg model checkpoint from: {leg_resume_path}")

        elif args_cli.options == "arm":
            print("agent_cfg.load_run",agent_cfg.load_run)
            print("agent_cfg.arm_load_checkpoint",agent_cfg.arm_load_checkpoint)
            arm_resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.arm_load_checkpoint)
            log_dir = os.path.dirname(arm_resume_path)
            print(f"[INFO]: Loading arm model checkpoint from: {arm_resume_path}")

        elif args_cli.options == "all":
            leg_resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.leg_load_checkpoint)
            arm_resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.arm_load_checkpoint)
            log_dir = os.path.dirname(leg_resume_path)
            print(f"[INFO]: Loading leg model checkpoint from: {leg_resume_path}")
            print(f"[INFO]: Loading arm model checkpoint from: {arm_resume_path}")


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if args_cli.options == "leg":
        ppo_runner.leg_load(leg_resume_path)
        leg_policy = ppo_runner.leg_get_inference_policy(device=env.unwrapped.device)
        leg_policy_nn = ppo_runner.leg_alg.policy
        
        leg_export_model_dir = os.path.join(os.path.dirname(leg_resume_path), "exported")
        export_policy_as_jit(leg_policy_nn, ppo_runner.leg_obs_normalizer, ppo_runner.arm_obs_normalizer, path=leg_export_model_dir, filename="leg_policy.pt")

    elif args_cli.options == "arm":
        ppo_runner.arm_load(arm_resume_path)
        arm_policy = ppo_runner.arm_get_inference_policy(device=env.unwrapped.device)
        arm_policy_nn = ppo_runner.arm_alg.policy
        
        arm_export_model_dir = os.path.join(os.path.dirname(arm_resume_path), "exported")
        export_policy_as_jit(arm_policy_nn, ppo_runner.arm_obs_normalizer, ppo_runner.leg_obs_normalizer, path=arm_export_model_dir, filename="arm_policy.pt")

    elif args_cli.options == "all":
        ppo_runner.leg_load(leg_resume_path)
        ppo_runner.arm_load(arm_resume_path)
        leg_policy = ppo_runner.leg_get_inference_policy(device=env.unwrapped.device)
        arm_policy = ppo_runner.arm_get_inference_policy(device=env.unwrapped.device)
        leg_policy_nn = ppo_runner.leg_alg.policy
        arm_policy_nn = ppo_runner.arm_alg.policy

        leg_export_model_dir = os.path.join(os.path.dirname(leg_resume_path), "exported")
        export_policy_as_jit(leg_policy_nn, ppo_runner.leg_obs_normalizer, ppo_runner.arm_obs_normalizer, path=leg_export_model_dir, filename="leg_policy.pt")
        arm_export_model_dir = os.path.join(os.path.dirname(arm_resume_path), "exported")
        export_policy_as_jit(arm_policy_nn, ppo_runner.arm_obs_normalizer, ppo_runner.leg_obs_normalizer, path=arm_export_model_dir, filename="arm_policy.pt")


    # # extract the neural network module
    # # we do this in a try-except to maintain backwards compatibility.
    # try:
    #     # version 2.3 onwards
    #     leg_policy_nn = ppo_runner.leg_alg.policy
    #     arm_policy_nn = ppo_runner.leg_alg.policy
    #     print("2.3")
    # except AttributeError:
    #     # version 2.2 and below
    #     # TODO:
    #     policy_nn = ppo_runner.leg_alg.actor_critic

    dt = env.unwrapped.step_dt
    # print("dt",dt)
    # open('data/joint_pos.txt', 'w').close()

    # reset environment
    leg_obs, arm_obs, _, _ = env.get_observations()
    timestep = 0
    leg_actions = torch.zeros(env.num_envs, env.num_leg_actions, device=env.device)
    arm_actions = torch.zeros(env.num_envs, env.num_arm_actions, device=env.device)
    timestart = time.time()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.options == "leg":
                leg_actions = leg_policy(leg_obs, arm_obs)
                
            elif args_cli.options =="arm":
                arm_actions = arm_policy(arm_obs, leg_obs)
                # arm_actions *= 0.0
                # arm_actions[0,0] = 0.01
                # arm_actions[0,1] = 0.02
                # arm_actions[0,2] = 0.03
                # arm_actions[0,3] = 0.04
                # arm_actions[0,4] = 0.05
                # arm_actions[0,5] = 0.06

                # # arm_actions[0,0] = 0.8
                # # arm_actions[0, 1] = 2.7
                # # arm_actions[0, 4] = -1.5
                # # arm_actions[0, 2] = 0.5
                # a = 10.0
                # arm_actions[0, 0] = a
                # arm_actions[1, 0] = -a

                # arm_actions[2, 1] = a
                # arm_actions[3, 1] = -a

                # arm_actions[4, 2] = a
                # arm_actions[5, 2] = -a
                # arm_actions[6, 3] = a
                # arm_actions[7, 3] = -a
                # arm_actions[8, 4] = a
                # arm_actions[9, 4] = -a
                # arm_actions[10, 5] = a
                # arm_actions[11, 5] = -a

            elif args_cli.options == "all":
                # print("arm_obs",arm_obs[:,:31])
                arm_actions = arm_policy(arm_obs, leg_obs)
                # print("arm_actions",arm_actions)
                leg_actions = leg_policy(leg_obs, arm_obs)
                # print("leg_obs",leg_obs[:,:45])
                # print("leg_actions",leg_actions)

            # print("leg_action", leg_actions.shape)
            # print("arm_action", arm_actions.shape)
            # print("arm_obs",arm_obs)
            # print("arm_actions",arm_actions)
            # leg_actions *= 0.0
            # arm_actions *= 0.0
            # arm_actions[:, 1] = 4.0
            # arm_actions[:, 2] = -1.0
            # arm_actions[:, 3] = -1.0


            # with open('data/joint_pos.txt', 'a') as f:
            #     tensor_cpu = (leg_obs[:,3:15]).detach().cpu() 

            #     # tensor_cpu = arm_obs[:,:6].detach().cpu() 
            #     tensor_str = np.array2string(tensor_cpu.numpy(), precision=4, separator=', ', suppress_small=True,max_line_width=np.inf)
            #     f.write(tensor_str + '\n')

            # actions *= 0.0
            # print("time", (time.time() - timestart))
            # if (time.time() - timestart) > 3.0:
            #     print("!")
            #     actions[:, 12:] = math.sin(2 * math.pi * (time.time() - timestart) / 2)
            #     # actions[:, :12] = math.sin(2 * math.pi * (time.time() - timestart) / 2)



            actions = torch.cat([leg_actions, arm_actions], dim = -1)
            # actions *= 0.0
            # print("act",actions.shape)
            # env stepping
            leg_obs, arm_obs, _, _, _, _, _, _ = env.step(actions)
            # print("leg_obs", leg_obs)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
