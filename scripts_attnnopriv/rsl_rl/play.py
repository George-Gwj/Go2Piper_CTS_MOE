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
parser.add_argument(
    "--record_isaaclab",
    "--record_video",
    action="store_true",
    default=False,
    dest="record_isaaclab",
    help="Record the Isaac Lab play visualization to a video.",
)
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=None,
    help="Directory to save play videos. Defaults to <checkpoint_dir>/videos/play.",
)
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
    "--max_play_steps",
    type=int,
    default=None,
    help="Stop play after this many simulation steps. Useful for short diagnostics.",
)
parser.add_argument(
    "--follow_robot",
    action="store_true",
    default=False,
    help="Make the visualization camera follow a robot body in the selected environment.",
)
parser.add_argument(
    "--follow_robot_env_index",
    type=int,
    default=0,
    help="Environment index for the follow camera.",
)
parser.add_argument(
    "--follow_robot_asset",
    type=str,
    default="robot",
    help="Scene asset name for the follow camera.",
)
parser.add_argument(
    "--follow_robot_body",
    type=str,
    default="base",
    help="Robot body name for the follow camera.",
)
parser.add_argument(
    "--follow_robot_eye",
    type=float,
    nargs=3,
    default=(-3.0, -3.0, 2.0),
    metavar=("X", "Y", "Z"),
    help="Camera eye offset for --follow_robot.",
)
parser.add_argument(
    "--follow_robot_lookat",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.5),
    metavar=("X", "Y", "Z"),
    help="Camera look-at offset for --follow_robot.",
)
parser.add_argument(
    "--print_router_interval",
    type=int,
    default=0,
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
parser.add_argument(
    "--collect_all_terrain_router_weights",
    action="store_true",
    default=False,
    help=(
        "Collect expert/router weight curves for one robot walking over the long all-terrain play scene. "
        "This forces one env, records robot0's continuous trajectory, disables live plotting, "
        "and stops after one episode."
    ),
)
parser.add_argument(
    "--log_gait_timing",
    action="store_true",
    default=False,
    help="Log foot contact, air time, contact time, and gait timing summaries during play.",
)
parser.add_argument(
    "--gait_log_interval",
    type=int,
    default=24,
    help="Print gait timing every N play steps when --log_gait_timing is enabled.",
)
parser.add_argument(
    "--gait_log_env_index",
    type=int,
    default=0,
    help="Environment index used for gait timing logs.",
)
parser.add_argument(
    "--gait_contact_threshold",
    type=float,
    default=1.0,
    help="Contact force threshold used to convert foot contact force to binary contact state.",
)
parser.add_argument(
    "--gait_output_dir",
    type=str,
    default=None,
    help="Directory to save gait timing CSV files. Defaults to <checkpoint_dir>/play_gait_timing.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.video = args_cli.video or args_cli.record_isaaclab
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.collect_all_terrain_router_weights:
    args_cli.num_envs = 1
    args_cli.plot_router_weights = True
    args_cli.live_plot_router = False
    args_cli.print_router_interval = 0

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from gymnasium import logger
from gymnasium import error
import os
import csv
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
from local_rsl_rl.wrappers import RslRlVecEnvWrapper, RslRlVecEnvWrapperSeparate

from Go2Piper_Attention.tasks.manager_based.go2piper_attention.config.agents.rsl_rl_ppo_cfg_moe import (
    Go2PiperCTSMoERunnerCfg,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Go2Piper_Attention.tasks  # noqa: F401


class RecordVideoSixTuple(gym.Wrapper):
    """Record videos for envs whose step API returns six values."""

    def __init__(
        self,
        env,
        video_folder: str,
        step_trigger,
        video_length: int,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
    ):
        super().__init__(env)
        if env.render_mode in {None, "human", "ansi"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with video recording.",
                "Initialize the environment with render_mode='rgb_array'.",
            )

        try:
            import moviepy  # noqa: F401
        except ImportError as exc:
            raise error.DependencyNotInstalled(
                'MoviePy is not installed, run `pip install "gymnasium[other]"`'
            ) from exc

        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)
        self.step_trigger = step_trigger
        self.video_length = int(video_length) if video_length != 0 else float("inf")
        self.name_prefix = name_prefix
        self.frames_per_sec = fps if fps is not None else self.metadata.get("render_fps", 30)
        self.disable_logger = disable_logger
        self.step_id = -1
        self.recording = False
        self.recorded_frames = []
        self._video_name = None

    def step(self, action):
        result = self.env.step(action)
        self.step_id += 1

        if self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) >= self.video_length:
                self.stop_recording()

        return result

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def close(self):
        if self.recording:
            self.stop_recording()
        return self.env.close()

    def start_recording(self, video_name: str):
        if self.recording:
            self.stop_recording()
        self.recording = True
        self.recorded_frames = []
        self._video_name = video_name

    def stop_recording(self):
        if not self.recording:
            return
        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            clip.write_videofile(path, logger=moviepy_logger)
            del clip

        self.recorded_frames = []
        self.recording = False
        self._video_name = None

    def _capture_frame(self):
        frame = self.env.render()
        if isinstance(frame, list):
            if not frame:
                return
            frame = frame[-1]
        if hasattr(frame, "shape"):
            self.recorded_frames.append(frame)
        else:
            self.stop_recording()
            logger.warn(
                f"Recording stopped: expected render frame with shape, got {type(frame)}."
            )


class GaitTimingLogger:
    """Record and summarize foot contact timing for one play environment."""

    def __init__(
        self,
        env,
        output_dir: str,
        env_index: int = 0,
        contact_threshold: float = 1.0,
        print_interval: int = 24,
    ):
        self.env = env
        self.output_dir = output_dir
        self.env_index = int(env_index)
        self.contact_threshold = float(contact_threshold)
        self.print_interval = int(print_interval)
        self.dt = float(env.unwrapped.step_dt)
        self.rows = []
        self.prev_contact = None
        self.current_phase_start = None
        self.stance_durations = None
        self.swing_durations = None
        self.sample_count = 0
        self.contact_counts = None
        self.transition_counts = None
        self.window_sample_count = 0
        self.window_contact_counts = None
        self.window_transition_counts = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        self.foot_ids, self.foot_names = self._resolve_foot_bodies()
        self.current_phase_start = [0.0 for _ in self.foot_ids]
        self.stance_durations = [[] for _ in self.foot_ids]
        self.swing_durations = [[] for _ in self.foot_ids]
        self.contact_counts = [0 for _ in self.foot_ids]
        self.transition_counts = [0 for _ in self.foot_ids]
        self.window_contact_counts = [0 for _ in self.foot_ids]
        self.window_transition_counts = [0 for _ in self.foot_ids]

        print(
            "[INFO] Gait timing logger enabled: "
            f"env_index={self.env_index}, feet={self.foot_names}, "
            f"dt={self.dt:.4f}s, output_dir={self.output_dir}"
        )

    def _resolve_foot_bodies(self):
        body_ids = self.contact_sensor.find_bodies(".*_foot")[0]
        body_names = getattr(self.contact_sensor, "body_names", None)
        body_ids = [int(body_id) for body_id in body_ids]
        if body_names is None:
            names = [f"foot_{idx}" for idx in range(len(body_ids))]
        else:
            names = [body_names[idx] for idx in body_ids]
        return list(body_ids), names

    def record(self, step: int, task_ids: torch.Tensor | None = None) -> None:
        forces = self.contact_sensor.data.net_forces_w[self.env_index, self.foot_ids, :].norm(dim=-1)
        contact = forces > self.contact_threshold
        contact_cpu = contact.detach().cpu().tolist()
        forces_cpu = forces.detach().cpu().tolist()
        air_time = self.contact_sensor.data.current_air_time[self.env_index, self.foot_ids]
        contact_time = self.contact_sensor.data.current_contact_time[self.env_index, self.foot_ids]
        air_cpu = air_time.detach().cpu().tolist()
        contact_time_cpu = contact_time.detach().cpu().tolist()

        t = step * self.dt
        self.sample_count += 1
        self.window_sample_count += 1
        for foot_idx, is_contact in enumerate(contact_cpu):
            if is_contact:
                self.contact_counts[foot_idx] += 1
                self.window_contact_counts[foot_idx] += 1

        if self.prev_contact is None:
            self.prev_contact = list(contact_cpu)
            self.current_phase_start = [t for _ in self.foot_ids]
        else:
            for foot_idx, is_contact in enumerate(contact_cpu):
                was_contact = self.prev_contact[foot_idx]
                if is_contact != was_contact:
                    duration = t - self.current_phase_start[foot_idx]
                    if was_contact:
                        self.stance_durations[foot_idx].append(duration)
                    else:
                        self.swing_durations[foot_idx].append(duration)
                    self.current_phase_start[foot_idx] = t
                    self.transition_counts[foot_idx] += 1
                    self.window_transition_counts[foot_idx] += 1
            self.prev_contact = list(contact_cpu)

        task_id = None
        if task_ids is not None:
            task_id = int(task_ids[self.env_index].detach().cpu().item())

        row = {"step": int(step), "time_s": t, "task_id": task_id}
        for foot_idx, foot_name in enumerate(self.foot_names):
            row[f"{foot_name}_contact"] = int(contact_cpu[foot_idx])
            row[f"{foot_name}_force"] = float(forces_cpu[foot_idx])
            row[f"{foot_name}_air_time"] = float(air_cpu[foot_idx])
            row[f"{foot_name}_contact_time"] = float(contact_time_cpu[foot_idx])
        self.rows.append(row)

        if self.print_interval > 0 and step % self.print_interval == 0:
            pattern = " ".join(
                f"{name}:{'C' if contact_cpu[idx] else 'A'}"
                for idx, name in enumerate(self.foot_names)
            )
            contact_ratio = " ".join(
                f"{name}:{self.window_contact_counts[idx] / max(self.window_sample_count, 1):.2f}"
                for idx, name in enumerate(self.foot_names)
            )
            transitions = " ".join(
                f"{name}:{self.window_transition_counts[idx]}"
                for idx, name in enumerate(self.foot_names)
            )
            air = " ".join(f"{name}:{air_cpu[idx]:.2f}" for idx, name in enumerate(self.foot_names))
            stance = " ".join(f"{name}:{contact_time_cpu[idx]:.2f}" for idx, name in enumerate(self.foot_names))
            task_text = f" task_id={task_id}" if task_id is not None else ""
            print(f"[gait step {step} t={t:.2f}s{task_text}] {pattern}")
            print(f"  window_contact_ratio: {contact_ratio}")
            print(f"  window_transitions: {transitions}")
            print(f"  air_time: {air}")
            print(f"  contact_time: {stance}")
            self.window_sample_count = 0
            self.window_contact_counts = [0 for _ in self.foot_ids]
            self.window_transition_counts = [0 for _ in self.foot_ids]

    def save(self) -> list[str]:
        if not self.rows:
            return []

        csv_path = os.path.join(self.output_dir, "gait_timing_steps.csv")
        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

        summary_path = os.path.join(self.output_dir, "gait_timing_summary.csv")
        summary_rows = []
        for foot_idx, foot_name in enumerate(self.foot_names):
            stance = self.stance_durations[foot_idx]
            swing = self.swing_durations[foot_idx]
            contact_ratio = self.contact_counts[foot_idx] / max(self.sample_count, 1)
            stance_mean = sum(stance) / len(stance) if stance else 0.0
            swing_mean = sum(swing) / len(swing) if swing else 0.0
            cycle_mean = stance_mean + swing_mean
            duty_factor = stance_mean / cycle_mean if cycle_mean > 0.0 else 0.0
            summary_rows.append(
                {
                    "foot": foot_name,
                    "sample_count": self.sample_count,
                    "contact_ratio": contact_ratio,
                    "air_ratio": 1.0 - contact_ratio,
                    "transition_count": self.transition_counts[foot_idx],
                    "stance_count": len(stance),
                    "swing_count": len(swing),
                    "mean_stance_s": stance_mean,
                    "mean_swing_s": swing_mean,
                    "mean_cycle_s": cycle_mean,
                    "duty_factor": duty_factor,
                }
            )

        with open(summary_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print("[INFO] Gait timing summary:")
        for row in summary_rows:
            print(
                f"  {row['foot']}: contact_ratio={row['contact_ratio']:.2f}, "
                f"air_ratio={row['air_ratio']:.2f}, transitions={row['transition_count']}, "
                f"stance={row['mean_stance_s']:.3f}s, "
                f"swing={row['mean_swing_s']:.3f}s, cycle={row['mean_cycle_s']:.3f}s, "
                f"duty={row['duty_factor']:.2f}"
            )
        return [csv_path, summary_path]


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
        router_weight = state_dict.get("leg_actor.router.4.weight")
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


def enable_all_terrain_router_collection(env_cfg) -> None:
    """Configure play env for collecting one robot's router weights over the long terrain."""
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = max(float(getattr(env_cfg, "episode_length_s", 0.0)), 60.0)

    task_cfg = getattr(env_cfg, "multi_task_rewards", None)
    if task_cfg is not None:
        task_cfg.play_long_terrain = True
        if hasattr(task_cfg, "play_long_axis"):
            task_cfg.play_long_axis = "y"

    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is not None and hasattr(curriculum_cfg, "terrain_levels"):
        curriculum_cfg.terrain_levels = None


def configure_follow_robot_viewer(env_cfg, args_cli) -> None:
    """Configure Isaac Lab viewer to follow a robot body."""
    viewer_cfg = getattr(env_cfg, "viewer", None)
    if viewer_cfg is None:
        raise ValueError("The selected environment config does not expose a viewer config.")
    if args_cli.follow_robot_env_index < 0:
        raise ValueError("--follow_robot_env_index must be >= 0.")

    viewer_cfg.origin_type = "asset_body"
    viewer_cfg.env_index = args_cli.follow_robot_env_index
    viewer_cfg.asset_name = args_cli.follow_robot_asset
    viewer_cfg.body_name = args_cli.follow_robot_body
    viewer_cfg.eye = list(args_cli.follow_robot_eye)
    viewer_cfg.lookat = list(args_cli.follow_robot_lookat)


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
    if args_cli.collect_all_terrain_router_weights:
        enable_all_terrain_router_collection(env_cfg)
    if args_cli.follow_robot:
        configure_follow_robot_viewer(env_cfg, args_cli)
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
    if args_cli.follow_robot:
        print(
            "[INFO]: Follow camera enabled for "
            f"env_index={args_cli.follow_robot_env_index}, "
            f"asset={args_cli.follow_robot_asset!r}, body={args_cli.follow_robot_body!r}."
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_output_dir
            if args_cli.video_output_dir is not None
            else os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = RecordVideoSixTuple(env, **video_kwargs)

    wrapper_cls = RslRlVecEnvWrapperSeparate if "Separate" in task_name else RslRlVecEnvWrapper
    env = wrapper_cls(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    task_names = getattr(env.unwrapped, "TASK_NAMES", runner.alg.policy.multi_critic.TASK_NAMES)
    expert_names = runner.alg.policy.moe_actor.expert_names
    if args_cli.print_router_interval > 0:
        print(f"[INFO] Printing MoE router weights every {args_cli.print_router_interval} steps.")

    plot_output_dir = (
        args_cli.plot_router_output_dir
        if args_cli.plot_router_output_dir is not None
        else os.path.join(
            log_dir,
            "all_terrain_router_plots"
            if args_cli.collect_all_terrain_router_weights
            else "play_router_plots",
        )
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
            max_points=max(2000, int(env.unwrapped.max_episode_length))
            if args_cli.collect_all_terrain_router_weights
            else 2000,
            trajectory_env_index=0 if args_cli.collect_all_terrain_router_weights else None,
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
    collection_stop_step = None
    gait_timing_logger = None
    if args_cli.log_gait_timing:
        gait_output_dir = (
            args_cli.gait_output_dir
            if args_cli.gait_output_dir is not None
            else os.path.join(log_dir, "play_gait_timing")
        )
        gait_timing_logger = GaitTimingLogger(
            env,
            output_dir=gait_output_dir,
            env_index=args_cli.gait_log_env_index,
            contact_threshold=args_cli.gait_contact_threshold,
            print_interval=args_cli.gait_log_interval,
        )
    if args_cli.collect_all_terrain_router_weights:
        collection_stop_step = int(env.unwrapped.max_episode_length)
        print(
            "[INFO] Collecting all-terrain robot0 router weights "
            f"for {collection_stop_step} steps ({env.unwrapped.cfg.episode_length_s:.1f}s)."
        )

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
            if gait_timing_logger is not None:
                gait_timing_logger.record(timestep, task_ids)

        if args_cli.video:
            timestep += 1
            if collection_stop_step is None and timestep == args_cli.video_length:
                break
        else:
            timestep += 1
        if collection_stop_step is not None and timestep >= collection_stop_step:
            print("[INFO] Finished all-terrain router weight collection episode.")
            break
        if args_cli.max_play_steps is not None and timestep >= args_cli.max_play_steps:
            print(f"[INFO] Reached --max_play_steps={args_cli.max_play_steps}.")
            break

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
    if gait_timing_logger is not None:
        saved_paths = gait_timing_logger.save()
        if saved_paths:
            print("[INFO] Saved gait timing logs:")
            for path in saved_paths:
                print(f"  - {path}")
        else:
            print("[WARN] No gait timing data was recorded.")


if __name__ == "__main__":
    main()
    simulation_app.close()
