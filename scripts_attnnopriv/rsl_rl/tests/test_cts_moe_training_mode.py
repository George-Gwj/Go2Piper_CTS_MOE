from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
RSL_RL_ROOT = REPO_ROOT / "scripts_attnnopriv/rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

from local_rsl_rl.algorithms import CTSMoEPPO
from local_rsl_rl.modules import StructureAwareCTSMoEPolicy


def _make_policy() -> StructureAwareCTSMoEPolicy:
    return StructureAwareCTSMoEPolicy(
        proprio_dim=8,
        privileged_dim=10,
        action_dim=4,
        latent_dim=16,
        height_channels=3,
        teacher_height_hidden_dims=(32,),
        teacher_privileged_hidden_dims=(32,),
        teacher_height_feature_dim=16,
        teacher_privileged_feature_dim=8,
        student_perception_type="depth",
        student_perception_channels=1,
        student_proprio_hidden_dims=(32,),
        student_proprio_feature_dim=8,
        student_depth_filters=(8, 16),
        student_depth_feature_dim=16,
        student_gru_hidden_dim=32,
        student_gru_num_layers=1,
        num_experts=2,
        num_tasks=2,
        expert_hidden_dims=(32,),
        router_hidden_dims=(16,),
        critic_hidden_dims=(32,),
        critic_head_hidden_dims=(16,),
        semantic_decoupled_teacher=False,
    )


def _make_obs(batch_size: int = 4):
    return {
        "proprio": torch.randn(batch_size, 8),
        "height_scan": torch.randn(batch_size, 3, 8, 8),
        "privileged_obs": torch.randn(batch_size, 10),
        "proprio_history": torch.randn(batch_size, 3, 8),
        "perception": torch.randn(batch_size, 1, 8, 8),
        "task_id": torch.randint(0, 2, (batch_size,)),
    }


def _run_rollout_and_update(alg: CTSMoEPPO, obs: dict[str, torch.Tensor]) -> dict:
    batch_size = obs["proprio"].shape[0]
    alg.init_storage(
        batch_size,
        2,
        list(obs["proprio"].shape[1:]),
        list(obs["height_scan"].shape[1:]),
        list(obs["privileged_obs"].shape[1:]),
        list(obs["proprio_history"].shape[1:]),
        list(obs["perception"].shape[1:]),
        [4],
    )

    for _ in range(2):
        actions = alg.act(**obs)
        rewards = torch.ones(batch_size)
        dones = torch.zeros(batch_size)
        alg.process_env_step(rewards, dones, {})

    alg.compute_returns(**obs)
    return alg.update()


def test_teacher_mode_skips_student_rollout_and_distillation():
    policy = _make_policy()
    alg = CTSMoEPPO(
        policy,
        device="cpu",
        training_mode="teacher",
        num_learning_epochs=1,
        num_mini_batches=1,
        use_popart=False,
    )
    obs = _make_obs()

    student_mask = alg._resolve_student_mask(obs["proprio"].shape[0], "cpu")
    assert not student_mask.any()

    loss_dict = _run_rollout_and_update(alg, obs)
    assert alg.training_mode == "teacher"
    assert loss_dict["distillation"] == 0.0
    assert loss_dict["student_rollout_ratio"] == 0.0


def test_mixed_mode_runs_distillation_when_student_mask_present():
    policy = _make_policy()
    alg = CTSMoEPPO(
        policy,
        device="cpu",
        training_mode="mixed",
        student_rollout_ratio=1.0,
        num_learning_epochs=1,
        num_mini_batches=1,
        use_popart=False,
    )
    obs = _make_obs()

    student_mask = alg._resolve_student_mask(obs["proprio"].shape[0], "cpu")
    assert student_mask.all()

    loss_dict = _run_rollout_and_update(alg, obs)
    assert alg.training_mode == "mixed"
    assert loss_dict["student_rollout_ratio"] == 1.0
