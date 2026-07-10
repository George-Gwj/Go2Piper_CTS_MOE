from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
RSL_RL_ROOT = REPO_ROOT / "scripts_attnnopriv/rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

from local_rsl_rl.modules import OrthogonalMoEActor, compute_orthogonality_metrics


def test_orthogonal_moe_actor_shapes_orthogonality_and_backward():
    torch.manual_seed(0)
    batch_size = 32
    latent_dim = 32
    proprio_dim = 66
    action_dim = 18
    num_experts = 4
    expert_feature_dim = 128

    actor = OrthogonalMoEActor(
        latent_dim=latent_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        num_experts=num_experts,
        expert_feature_dim=expert_feature_dim,
        expert_hidden_dims=(64,),
        router_hidden_dims=(32,),
        action_head_hidden_dims=(64,),
        orthogonal_mode="gram_schmidt",
    )
    z = torch.randn(batch_size, latent_dim)
    proprio = torch.randn(batch_size, proprio_dim)

    action_mean, gate_weights, expert_actions, _router_logits, extras = actor(z, proprio)
    U = extras["expert_features_raw"]
    V = extras["expert_features_orth"]

    assert action_mean.shape == (batch_size, action_dim)
    assert gate_weights.shape == (batch_size, num_experts)
    assert expert_actions.shape == (batch_size, num_experts, action_dim)
    assert U.shape == (batch_size, num_experts, expert_feature_dim)
    assert V.shape == (batch_size, num_experts, expert_feature_dim)

    gram = torch.matmul(V, V.transpose(-1, -2))
    diag = gram.diagonal(dim1=-2, dim2=-1)
    off_diag_mask = ~torch.eye(num_experts, dtype=torch.bool)
    off_diag_mean_abs = gram[:, off_diag_mask].abs().mean()
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4, rtol=1e-4)
    assert off_diag_mean_abs < 1e-3

    metrics = compute_orthogonality_metrics(V, gate_weights)
    assert "actor/orth_offdiag_mean_abs" in metrics
    assert "actor/gate_entropy" in metrics

    loss = action_mean.mean()
    loss.backward()
    grads = [param.grad for param in actor.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
