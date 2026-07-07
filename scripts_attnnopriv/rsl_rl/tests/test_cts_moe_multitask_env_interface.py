from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def load_manager_env_class():
    repo_root = Path(__file__).resolve().parents[3]
    manager_env_path = repo_root / "source/Go2Piper_Attention/Go2Piper_Attention/env/manager_env.py"

    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_envs_module = types.ModuleType("isaaclab.envs")

    class ManagerBasedRLEnv:
        def _reset_idx(self, env_ids):
            self._last_base_reset_env_ids = env_ids

    isaaclab_envs_module.ManagerBasedRLEnv = ManagerBasedRLEnv
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.envs"] = isaaclab_envs_module

    fake_package = types.ModuleType("fake_manager_env_package")
    fake_package.__path__ = []
    sys.modules["fake_manager_env_package"] = fake_package
    sys.modules["fake_manager_env_package.local_manager"] = types.ModuleType("fake_manager_env_package.local_manager")

    spec = importlib.util.spec_from_file_location("fake_manager_env_package.manager_env", manager_env_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ManagerRLEnv


ManagerRLEnv = load_manager_env_class()


class FakeRobot:
    def __init__(self, num_envs: int, device: str):
        self.data = SimpleNamespace(root_pos_w=torch.zeros(num_envs, 3, device=device))


class FakeRewardManager:
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device

    def compute_grouped_by_task_marker(self, dt: float):
        grouped_rewards = {
            "common": torch.ones(self.num_envs, device=self.device) * 0.2,
            "box_avoidance": torch.ones(self.num_envs, device=self.device) * 1.0,
            "under_table": torch.ones(self.num_envs, device=self.device) * 2.0,
            "stair_up": torch.ones(self.num_envs, device=self.device) * 3.0,
            "flat": torch.ones(self.num_envs, device=self.device) * 4.0,
        }
        grouped_logs = {
            "common": {"tracking": grouped_rewards["common"]},
            "box_avoidance": {"clearance": grouped_rewards["box_avoidance"]},
            "under_table": {"clearance": grouped_rewards["under_table"]},
            "stair_up": {"progress": grouped_rewards["stair_up"]},
            "flat": {"progress": grouped_rewards["flat"]},
        }
        return grouped_rewards, grouped_logs


def make_fake_env(
    num_envs: int = 17,
    fixed_task_assignment: bool = True,
    fixed_task_id: int | None = None,
    task_sampling_weights: list[float] | None = None,
):
    env = ManagerRLEnv.__new__(ManagerRLEnv)
    env.num_envs = num_envs
    env.device = "cpu"
    env.extras = {}
    env._cts_moe_enabled = True
    env._cts_moe_reward_log = {}
    env.cfg = SimpleNamespace(
        multi_task_rewards=SimpleNamespace(
            alive_weight=0.1,
            fixed_task_assignment=fixed_task_assignment,
            fixed_task_id=fixed_task_id,
            task_sampling_weights=task_sampling_weights,
        )
    )
    env.robot = FakeRobot(num_envs, env.device)
    env.reward_manager = FakeRewardManager(num_envs, env.device)
    env.step_dt = 0.02
    env.prev_base_pos = torch.zeros(num_envs, 3, device=env.device)
    return env


def test_fixed_task_assignment_and_rewards():
    env = make_fake_env(num_envs=17)
    env._assign_env_tasks()
    env._setup_task_scenes()

    assert env.task_id.shape == (17,)
    assert env.task_id.dtype == torch.long
    assert int(env.task_id.min()) >= 0
    assert int(env.task_id.max()) < env.NUM_TASKS
    assert torch.bincount(env.task_id, minlength=env.NUM_TASKS).tolist() == [5, 4, 4, 4]

    task_id_before_reset = env.task_id.clone()
    env._reset_idx(torch.tensor([0, 3, 8, 16], dtype=torch.long))
    assert torch.equal(env.task_id, task_id_before_reset)

    reward = env._get_rewards()
    assert reward.shape == (17,)
    expected_reward = torch.ones(17) * 0.3
    expected_reward[env.task_id == env.TASK_BOX_AVOIDANCE] += 1.0
    expected_reward[env.task_id == env.TASK_UNDER_TABLE] += 2.0
    expected_reward[env.task_id == env.TASK_STAIR_UP] += 3.0
    expected_reward[env.task_id == env.TASK_FLAT] += 4.0
    assert torch.allclose(reward, expected_reward)

    required_log_keys = {
        "rew/common/alive",
        "rew/common/tracking",
        "rew/common/marked_total",
        "rew/box/clearance",
        "rew/box/placeholder",
        "rew/under_table/clearance",
        "rew/under_table/placeholder",
        "rew/stair_up/progress",
        "rew/stair_up/placeholder",
        "rew/flat/progress",
        "rew/flat/placeholder",
        "task/num_box",
        "task/num_under_table",
        "task/num_stair_up",
        "task/num_flat",
    }
    assert required_log_keys.issubset(env.extras["log"].keys())


def test_fixed_task_id():
    env = make_fake_env(num_envs=8, fixed_task_id=ManagerRLEnv.TASK_STAIR_UP)
    env._assign_env_tasks()
    assert torch.all(env.task_id == ManagerRLEnv.TASK_STAIR_UP)
    assert env.mask_stair_up.sum().item() == 8


if __name__ == "__main__":
    test_fixed_task_assignment_and_rewards()
    test_fixed_task_id()
    print("CTS-MoE multitask env interface dry-run passed")
