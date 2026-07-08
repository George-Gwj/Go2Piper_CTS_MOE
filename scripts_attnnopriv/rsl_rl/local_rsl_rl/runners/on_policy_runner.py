from __future__ import annotations

from collections import deque
import inspect
import os
import statistics
import time

import torch

import local_rsl_rl
from local_rsl_rl.algorithms import CTSMoEPPO
from local_rsl_rl.env import VecEnv
from local_rsl_rl.modules import StructureAwareCTSMoEPolicy
from local_rsl_rl.utils import store_code_state


class OnPolicyRunner:
    """CTS-MoE-only on-policy runner."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.env = env
        self.device = device

        self.policy_cfg = dict(train_cfg["policy"])
        self.alg_cfg = dict(train_cfg["algorithm"])
        policy_class_name = self.policy_cfg.pop("class_name", "StructureAwareCTSMoEPolicy")
        alg_class_name = self.alg_cfg.pop("class_name", "CTSMoEPPO")
        if policy_class_name != "StructureAwareCTSMoEPolicy":
            raise ValueError(f"Unsupported CTS-MoE policy class: {policy_class_name}")
        if alg_class_name != "CTSMoEPPO":
            raise ValueError(f"Unsupported CTS-MoE algorithm class: {alg_class_name}")
        self.alg_cfg = self._filter_constructor_cfg(CTSMoEPPO, self.alg_cfg)
        self.alg_cfg = self._apply_training_mode_override(self.alg_cfg)

        obs = self._move_obs_to_device(self.env.get_cts_moe_observations())
        self._sync_policy_dims_from_obs(obs)

        self.policy = StructureAwareCTSMoEPolicy(**self.policy_cfg).to(self.device)
        self.alg = CTSMoEPPO(self.policy, device=self.device, **self.alg_cfg)
        self.training_mode = self.alg.training_mode

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg.get("empirical_normalization", False)
        if self.empirical_normalization:
            raise NotImplementedError("CTS-MoE structured observation normalization is not implemented.")

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            list(obs["proprio"].shape[1:]),
            list(obs["height_scan"].shape[1:]),
            list(obs["privileged_obs"].shape[1:]),
            list(obs["proprio_history"].shape[1:]),
            list(obs["perception"].shape[1:]),
            [self.env.num_actions],
        )

        self.is_distributed = False
        self.disable_logs = self.is_distributed
        self.log_dir = log_dir
        self.writer = None
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.git_status_repos = [local_rsl_rl.__file__]

    def _filter_constructor_cfg(self, cls, cfg: dict) -> dict:
        signature = inspect.signature(cls.__init__)
        valid_keys = set(signature.parameters) - {"self", "policy", "device"}
        return {key: value for key, value in cfg.items() if key in valid_keys}

    def _apply_training_mode_override(self, alg_cfg: dict) -> dict:
        options = self.cfg.get("options")
        if options in ("teacher", "mix", "mixed"):
            alg_cfg["training_mode"] = "teacher" if options == "teacher" else "mixed"
        return alg_cfg

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self._init_writer()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self._move_obs_to_device(self.env.get_cts_moe_observations())
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(**obs)
                    next_obs, rewards, dones, infos = self.env.step_cts_moe(actions.to(self.env.device))
                    obs = self._move_obs_to_device(next_obs)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        done_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[done_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[done_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[done_ids] = 0
                        cur_episode_length[done_ids] = 0

                collection_time = time.time() - start
                learn_start = time.time()
                self.alg.compute_returns(**obs)

            loss_dict = self.alg.update()
            learn_time = time.time() - learn_start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self._log_cts_moe(
                    it=it,
                    start_iter=start_iter,
                    tot_iter=tot_iter,
                    num_learning_iterations=num_learning_iterations,
                    collection_time=collection_time,
                    learn_time=learn_time,
                    loss_dict=loss_dict,
                    ep_infos=ep_infos,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                )
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"CTSMoE_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and self.log_dir is not None and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"CTSMoE_{self.current_learning_iteration}.pt"))

    def _sync_policy_dims_from_obs(self, obs: dict[str, torch.Tensor]):
        self.policy_cfg["proprio_dim"] = obs["proprio"].shape[-1]
        self.policy_cfg["privileged_dim"] = obs["privileged_obs"].shape[-1]
        self.policy_cfg["action_dim"] = self.env.num_actions
        if obs["height_scan"].dim() == 4:
            self.policy_cfg["height_channels"] = obs["height_scan"].shape[1]
        if obs["perception"].dim() == 4:
            self.policy_cfg["student_perception_channels"] = obs["perception"].shape[1]

    def _move_obs_to_device(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in obs.items()}

    def _init_writer(self):
        if self.log_dir is None or self.writer is not None or self.disable_logs:
            return
        if self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        elif self.logger_type == "wandb":
            from local_rsl_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "neptune":
            from local_rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        else:
            raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def _log_cts_moe(
        self,
        it: int,
        start_iter: int,
        tot_iter: int,
        num_learning_iterations: int,
        collection_time: float,
        learn_time: float,
        loss_dict: dict,
        ep_infos: list,
        rewbuffer: deque,
        lenbuffer: deque,
        width: int = 80,
        pad: int = 35,
    ):
        collection_size = self.num_steps_per_env * self.env.num_envs
        self.tot_timesteps += collection_size
        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time
        fps = int(collection_size / max(iteration_time, 1e-6))

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)
        self.writer.add_scalar("Policy/mean_noise_std", self.alg.policy.action_std.mean().item(), it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collection_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        ep_string = self._log_episode_infos(ep_infos, it, pad)
        if len(rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)

        title = f" \033[1m CTS-MoE ({self.training_mode}) learning iteration {it}/{tot_iter} \033[0m "
        log_string = (
            f"{'#' * width}\n"
            f"{title.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
            f"{'Mean action noise std:':>{pad}} {self.alg.policy.action_std.mean().item():.2f}\n"
        )
        for key, value in loss_dict.items():
            log_string += f"{f'Mean {key}:':>{pad}} {value:.4f}\n"
        if len(rewbuffer) > 0:
            log_string += f"{'Mean reward:':>{pad}} {statistics.mean(rewbuffer):.2f}\n"
            log_string += f"{'Mean episode length:':>{pad}} {statistics.mean(lenbuffer):.2f}\n"
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"
            f"{'ETA:':>{pad}} "
            f"{time.strftime('%H:%M:%S', time.gmtime(self.tot_time / (it - start_iter + 1) * (start_iter + num_learning_iterations - it)))}\n"
        )
        print(log_string)

    def _log_episode_infos(self, ep_infos: list, it: int, pad: int) -> str:
        if not ep_infos:
            return ""
        ep_string = ""
        for key in ep_infos[0]:
            infotensor = torch.tensor([], device=self.device)
            for ep_info in ep_infos:
                if key not in ep_info:
                    continue
                value = ep_info[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.Tensor([value])
                if len(value.shape) == 0:
                    value = value.unsqueeze(0)
                infotensor = torch.cat((infotensor, value.to(self.device)))
            if infotensor.numel() == 0:
                continue
            mean_value = torch.mean(infotensor)
            self.writer.add_scalar(key if "/" in key else "Episode/" + key, mean_value, it)
            ep_string += f"{f'{key}:':>{pad}} {mean_value:.4f}\n"
        return ep_string

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "student_optimizer_state_dict": self.alg.student_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg.popart is not None:
            saved_dict["popart_state_dict"] = self.alg.popart.state_dict()
        torch.save(saved_dict, path)
        if self.writer is not None and self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "student_optimizer_state_dict" in loaded_dict:
                self.alg.student_optimizer.load_state_dict(loaded_dict["student_optimizer_state_dict"])
        if self.alg.popart is not None and "popart_state_dict" in loaded_dict:
            self.alg.popart.load_state_dict(loaded_dict["popart_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos")

    def train_mode(self):
        self.alg.policy.train()

    def eval_mode(self):
        self.alg.policy.eval()

    def get_inference_policy(self, device=None, inference_mode: str | None = None):
        """Return a deterministic CTS-MoE policy callable for play/eval."""
        self.eval_mode()
        policy = self.alg.policy
        run_device = self.device if device is None else device
        if inference_mode is None:
            inference_mode = "teacher" if self.training_mode == "teacher" else "teacher"
        if inference_mode not in ("teacher", "student"):
            raise ValueError(f"inference_mode must be 'teacher' or 'student', got {inference_mode!r}")

        def _move_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {key: value.to(run_device) for key, value in obs.items()}

        if inference_mode == "teacher":

            def act(obs: dict[str, torch.Tensor]) -> torch.Tensor:
                obs = _move_obs(obs)
                out = policy(
                    mode="teacher",
                    proprio=obs["proprio"],
                    task_id=obs["task_id"],
                    height_scan=obs["height_scan"],
                    privileged_obs=obs["privileged_obs"],
                )
                return out["action_mean"]

            return act

        def act(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            obs = _move_obs(obs)
            out = policy(
                mode="student",
                proprio=obs["proprio"],
                task_id=obs["task_id"],
                proprio_history=obs["proprio_history"],
                perception=obs["perception"],
            )
            return out["action_mean"]

        return act

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
