# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one CTS-MoE episode and report per-terrain success rates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate CTS-MoE policy success rate by terrain.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--routing",
    type=str,
    default="soft",
    choices=("soft", "one_hot"),
    help="MoE routing at eval time: learned soft mixture or task-id one-hot expert selection.",
)
parser.add_argument(
    "--success_min_episode_ratio",
    type=float,
    default=0.99,
    help="Minimum fraction of max episode length required for a survival success.",
)
parser.add_argument("--vx_error_threshold", type=float, default=0.25, help="Mean abs vx tracking error threshold.")
parser.add_argument("--vy_error_threshold", type=float, default=0.25, help="Mean abs vy tracking error threshold.")
parser.add_argument("--wz_error_threshold", type=float, default=0.35, help="Mean abs yaw-rate tracking error threshold.")
parser.add_argument(
    "--progress_threshold",
    type=float,
    default=None,
    help="Optional minimum world-x progress for tracking success. Disabled by default.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save results. Defaults to the checkpoint/policy directory.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from local_rsl_rl.runners import OnPolicyRunner
from local_rsl_rl.wrappers import RslRlVecEnvWrapper

from Go2Piper_Attention.tasks.manager_based.go2piper_attention.config.agents.rsl_rl_ppo_cfg_moe import (
    Go2PiperCTSMoERunnerCfg,
)

import isaaclab_tasks  # noqa: F401
import Go2Piper_Attention.tasks  # noqa: F401


def is_cts_moe_task(task_name: str) -> bool:
    return "CTS-MoE" in task_name


def resolve_inference_mode(options: str | None) -> str:
    if options in (None, "teacher", "mix", "mixed"):
        return "teacher"
    if options == "student":
        return "student"
    raise ValueError(f"CTS-MoE eval expects --options teacher or --options student. Got {options!r}.")


def resolve_checkpoint_path(args_cli, agent_cfg, log_root_path: str) -> str:
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def sync_policy_experts_from_checkpoint(agent_cfg, checkpoint_path: str):
    """Match eval policy expert count to the checkpoint actor architecture."""
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
        "[INFO]: Adjusting eval policy num_experts "
        f"from {current_num_experts} to {checkpoint_num_experts} to match checkpoint."
    )
    policy_cfg.num_experts = checkpoint_num_experts
    policy_cfg.expert_names = [f"expert_{idx}" for idx in range(checkpoint_num_experts)]


def get_base_command(env) -> torch.Tensor:
    command = env.unwrapped.command_manager.get_command("base_velocity")
    if command.shape[-1] < 3:
        raise ValueError(f"base_velocity command must have at least 3 dims, got {tuple(command.shape)}")
    return command[:, :3]


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def write_results(output_prefix: str, summary: dict, rows: list[dict]):
    json_path = f"{output_prefix}_success_eval.json"
    csv_path = f"{output_prefix}_success_eval.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        "terrain",
        "task_id",
        "count",
        "survival_success",
        "tracking_success",
        "survival_success_rate",
        "tracking_success_rate",
        "terminated_rate",
        "timeout_rate",
        "avg_episode_length",
        "avg_abs_vx_error",
        "avg_abs_vy_error",
        "avg_abs_wz_error",
        "avg_progress_x",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: Go2PiperCTSMoERunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    if not is_cts_moe_task(task_name):
        raise ValueError(f"Task {task_name!r} is not a CTS-MoE task.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = resolve_checkpoint_path(args_cli, agent_cfg, log_root_path)
    sync_policy_experts_from_checkpoint(agent_cfg, resume_path)
    inference_mode = resolve_inference_mode(args_cli.options)
    checkpoint_dir = os.path.dirname(resume_path)
    policy_name = os.path.splitext(os.path.basename(resume_path))[0]
    output_dir = args_cli.output_dir if args_cli.output_dir is not None else checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, policy_name)

    print(f"[INFO]: Loading CTS-MoE checkpoint from: {resume_path}")
    print(f"[INFO]: Using inference_mode={inference_mode}, routing={args_cli.routing}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    obs, _ = env.reset()
    obs = {key: value.to(agent_cfg.device) for key, value in obs.items()}
    num_envs = env.num_envs
    device = env.device
    max_episode_length = int(env.max_episode_length)
    min_success_steps = int(round(args_cli.success_min_episode_ratio * max_episode_length))

    initial_task_ids = obs["task_id"].long().to(device).clone()
    initial_root_pos = env.unwrapped.robot.data.root_pos_w[:, :3].clone()
    last_root_pos = initial_root_pos.clone()

    sum_abs_vx_err = torch.zeros(num_envs, device=device)
    sum_abs_vy_err = torch.zeros(num_envs, device=device)
    sum_abs_wz_err = torch.zeros(num_envs, device=device)
    episode_steps = torch.zeros(num_envs, device=device)
    terminated_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    timeout_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    done_once = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for _ in range(max_episode_length):
        with torch.inference_mode():
            actions, _ = runner.infer_step(obs, inference_mode=inference_mode, routing_mode=args_cli.routing)
            actions = actions.to(device)

            active = ~done_once
            last_root_pos[active] = env.unwrapped.robot.data.root_pos_w[active, :3]
            commands = get_base_command(env)
            base_lin_vel = env.unwrapped.robot.data.root_lin_vel_b
            base_ang_vel = env.unwrapped.robot.data.root_ang_vel_b
            sum_abs_vx_err[active] += torch.abs(base_lin_vel[active, 0] - commands[active, 0])
            sum_abs_vy_err[active] += torch.abs(base_lin_vel[active, 1] - commands[active, 1])
            sum_abs_wz_err[active] += torch.abs(base_ang_vel[active, 2] - commands[active, 2])
            episode_steps[active] += 1.0

            obs, _, dones, _ = env.step_cts_moe(actions)
            obs = {key: value.to(agent_cfg.device) for key, value in obs.items()}

            terminated = env.unwrapped.reset_terminated.bool()
            timeout = env.unwrapped.reset_time_outs.bool()
            active_done = active & dones.bool()
            terminated_once |= active & terminated
            timeout_once |= active & timeout
            done_once |= active_done

        if bool(done_once.all()):
            break

    progress_x = last_root_pos[:, 0] - initial_root_pos[:, 0]
    denom = torch.clamp(episode_steps, min=1.0)
    mean_abs_vx_err = sum_abs_vx_err / denom
    mean_abs_vy_err = sum_abs_vy_err / denom
    mean_abs_wz_err = sum_abs_wz_err / denom

    survival_success = (~terminated_once) & (episode_steps >= min_success_steps)
    tracking_success = (
        survival_success
        & (mean_abs_vx_err <= args_cli.vx_error_threshold)
        & (mean_abs_vy_err <= args_cli.vy_error_threshold)
        & (mean_abs_wz_err <= args_cli.wz_error_threshold)
    )
    if args_cli.progress_threshold is not None:
        tracking_success &= progress_x >= args_cli.progress_threshold

    task_names = getattr(env.unwrapped, "TASK_NAMES", tuple(f"task_{idx}" for idx in range(5)))
    rows = []
    per_terrain = {}
    for task_id, terrain_name in enumerate(task_names):
        mask = initial_task_ids == task_id
        count = int(mask.sum().item())
        survival_count = int(survival_success[mask].sum().item()) if count > 0 else 0
        tracking_count = int(tracking_success[mask].sum().item()) if count > 0 else 0
        terminated_count = int(terminated_once[mask].sum().item()) if count > 0 else 0
        timeout_count = int(timeout_once[mask].sum().item()) if count > 0 else 0
        row = {
            "terrain": terrain_name,
            "task_id": task_id,
            "count": count,
            "survival_success": survival_count,
            "tracking_success": tracking_count,
            "survival_success_rate": safe_rate(survival_count, count),
            "tracking_success_rate": safe_rate(tracking_count, count),
            "terminated_rate": safe_rate(terminated_count, count),
            "timeout_rate": safe_rate(timeout_count, count),
            "avg_episode_length": float(episode_steps[mask].mean().item()) if count > 0 else 0.0,
            "avg_abs_vx_error": float(mean_abs_vx_err[mask].mean().item()) if count > 0 else 0.0,
            "avg_abs_vy_error": float(mean_abs_vy_err[mask].mean().item()) if count > 0 else 0.0,
            "avg_abs_wz_error": float(mean_abs_wz_err[mask].mean().item()) if count > 0 else 0.0,
            "avg_progress_x": float(progress_x[mask].mean().item()) if count > 0 else 0.0,
        }
        rows.append(row)
        per_terrain[terrain_name] = row

    total_count = num_envs
    total_survival = int(survival_success.sum().item())
    total_tracking = int(tracking_success.sum().item())
    summary = {
        "policy": os.path.basename(resume_path),
        "checkpoint_path": resume_path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "num_envs": num_envs,
        "max_episode_length": max_episode_length,
        "min_success_steps": min_success_steps,
        "inference_mode": inference_mode,
        "routing": args_cli.routing,
        "thresholds": {
            "success_min_episode_ratio": args_cli.success_min_episode_ratio,
            "vx_error": args_cli.vx_error_threshold,
            "vy_error": args_cli.vy_error_threshold,
            "wz_error": args_cli.wz_error_threshold,
            "progress": args_cli.progress_threshold,
        },
        "overall": {
            "count": total_count,
            "survival_success": total_survival,
            "tracking_success": total_tracking,
            "survival_success_rate": safe_rate(total_survival, total_count),
            "tracking_success_rate": safe_rate(total_tracking, total_count),
            "avg_episode_length": float(episode_steps.mean().item()),
            "avg_abs_vx_error": float(mean_abs_vx_err.mean().item()),
            "avg_abs_vy_error": float(mean_abs_vy_err.mean().item()),
            "avg_abs_wz_error": float(mean_abs_wz_err.mean().item()),
            "avg_progress_x": float(progress_x.mean().item()),
        },
        "per_terrain": per_terrain,
    }

    json_path, csv_path = write_results(output_prefix, summary, rows)
    print(f"[INFO]: Saved JSON results to: {json_path}")
    print(f"[INFO]: Saved CSV results to: {csv_path}")
    print(json.dumps(summary["overall"], indent=2))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
