# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a CTS-MoE checkpoint with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import argparse
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a CTS-MoE RL agent checkpoint with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--print_router_interval",
    type=int,
    default=24,
    help="Print MoE router weights every N play steps. Set 0 to disable.",
)
parser.add_argument(
    "--routing",
    type=str,
    default="soft",
    choices=("soft", "one_hot"),
    help="MoE routing at play time: learned soft mixture or task-id one-hot expert selection.",
)
parser.add_argument(
    "--plot_router_weights",
    action="store_true",
    default=True,
    help="Record per-task MoE router weight curves during play.",
)
parser.add_argument(
    "--no_plot_router_weights",
    action="store_false",
    dest="plot_router_weights",
    help="Disable per-task MoE router weight curve recording.",
)
parser.add_argument(
    "--plot_router_interval",
    type=int,
    default=1,
    help="Sample router weights every N play steps for curve plotting.",
)
parser.add_argument(
    "--plot_router_output_dir",
    type=str,
    default=None,
    help="Directory to save router weight plots/csv. Defaults to <checkpoint_dir>/play_router_plots.",
)
parser.add_argument(
    "--live_plot_router",
    action="store_true",
    default=True,
    help="Show a live matplotlib window with per-task MoE router weight curves during play.",
)
parser.add_argument(
    "--no_live_plot_router",
    action="store_false",
    dest="live_plot_router",
    help="Disable live router weight plotting; only save curves when play exits.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from local_rsl_rl.runners import OnPolicyRunner
from local_rsl_rl.utils.play_router_plotter import PlayRouterWeightLogger

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from local_rsl_rl.wrappers import RslRlVecEnvWrapper

from Go2Piper_Attention.tasks.manager_based.go2piper_attention.config.agents.rsl_rl_ppo_cfg_moe import (
    Go2PiperCTSMoERunnerCfg,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Go2Piper_Attention.tasks  # noqa: F401


def is_cts_moe_task(task_name: str) -> bool:
    return "CTS-MoE" in task_name


def resolve_inference_mode(options: str | None) -> str:
    if options in (None, "teacher", "mix", "mixed"):
        return "teacher"
    if options == "student":
        return "student"
    raise ValueError(
        "CTS-MoE play expects --options teacher or --options student. "
        f"Got {options!r}."
    )


def resolve_checkpoint_path(args_cli, agent_cfg, log_root_path: str) -> str:
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def sync_policy_experts_from_checkpoint(agent_cfg, checkpoint_path: str):
    """Match play policy expert count to the checkpoint actor architecture."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    router_weight = state_dict.get("moe_actor.router.4.weight")
    if router_weight is None:
        return

    checkpoint_num_experts = int(router_weight.shape[0])
    policy_cfg = agent_cfg.policy
    current_num_experts = int(getattr(policy_cfg, "num_experts", checkpoint_num_experts))
    if current_num_experts == checkpoint_num_experts:
        return

    print(
        "[INFO]: Adjusting play policy num_experts "
        f"from {current_num_experts} to {checkpoint_num_experts} to match checkpoint."
    )
    policy_cfg.num_experts = checkpoint_num_experts
    policy_cfg.expert_names = [f"expert_{idx}" for idx in range(checkpoint_num_experts)]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: Go2PiperCTSMoERunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    if not is_cts_moe_task(task_name):
        raise ValueError(
            f"Task {task_name!r} is not a CTS-MoE play task. "
            "Use Go2Piper-Attention-CTS-MoE-Teacher-Play or Go2Piper-Attention-CTS-MoE-Play."
        )

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = resolve_checkpoint_path(args_cli, agent_cfg, log_root_path)
    sync_policy_experts_from_checkpoint(agent_cfg, resume_path)
    inference_mode = resolve_inference_mode(args_cli.options)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO]: Loading CTS-MoE checkpoint from: {resume_path}")
    print(f"[INFO]: Using inference_mode={inference_mode}")
    print(f"[INFO]: Using routing_mode={args_cli.routing}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    task_names = getattr(env.unwrapped, "TASK_NAMES", runner.alg.policy.multi_critic.TASK_NAMES)
    expert_names = runner.alg.policy.moe_actor.expert_names
    if args_cli.print_router_interval > 0:
        print(f"[INFO] Printing MoE router weights every {args_cli.print_router_interval} steps.")

    plot_output_dir = (
        args_cli.plot_router_output_dir
        if args_cli.plot_router_output_dir is not None
        else os.path.join(log_dir, "play_router_plots")
    )
    router_plot_logger = None
    if args_cli.plot_router_weights:
        router_plot_logger = PlayRouterWeightLogger(
            task_names=task_names,
            expert_names=expert_names,
            output_dir=plot_output_dir,
            sample_interval=args_cli.plot_router_interval,
            live_plot=args_cli.live_plot_router,
            routing_mode=args_cli.routing,
            inference_mode=inference_mode,
        )
        if args_cli.live_plot_router:
            print(
                f"[INFO] Live router weight plot enabled "
                f"(refresh every {args_cli.plot_router_interval} steps)."
            )
        print(
            f"[INFO] Recording per-task router weight curves every "
            f"{args_cli.plot_router_interval} steps to: {plot_output_dir}"
        )

    dt = env.unwrapped.step_dt
    obs, _ = env.reset()
    obs = {key: value.to(agent_cfg.device) for key, value in obs.items()}
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            task_ids = obs["task_id"]
            actions, router_weights = runner.infer_step(
                obs,
                inference_mode=inference_mode,
                routing_mode=args_cli.routing,
            )
            actions = actions.to(env.device)
            obs, _, _, _ = env.step_cts_moe(actions)
            obs = {key: value.to(agent_cfg.device) for key, value in obs.items()}

            if args_cli.print_router_interval > 0 and timestep % args_cli.print_router_interval == 0:
                router_entries = OnPolicyRunner.aggregate_router_weights_by_task(
                    router_weights,
                    task_ids,
                    task_names,
                    expert_names,
                )
                print(f"[play step {timestep}] routing={args_cli.routing}")
                print(OnPolicyRunner.format_router_weight_table(router_entries))

            if router_plot_logger is not None:
                router_plot_logger.maybe_record(timestep, router_weights, task_ids)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

    if router_plot_logger is not None:
        router_plot_logger.close()
        saved_paths = router_plot_logger.save()
        if saved_paths:
            print("[INFO] Saved MoE router weight curves:")
            for path in saved_paths:
                print(f"  - {path}")
        else:
            print("[WARN] No router weight data was recorded for plotting.")


if __name__ == "__main__":
    main()
    simulation_app.close()
