# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .amp_discriminator import AMPDiscriminator
from .amp_motion_dataset import AMPMotionDataset
from .hybrid_leg_arm_cts_moe import ArmCritic, ArmMLPActor, HybridLegArmCTSMoEPolicy
from .normalizer import EmpiricalNormalization
from .structure_aware_cts_moe import (
    MoEActor,
    OrthogonalMoEActor,
    SparseMultiCritic,
    StructureAwareCTSMoEPolicy,
    StructureAwareDualRouterCTSMoEPolicy,
    StudentEncoder,
    TeacherEncoder,
    DualRouterHistoryOrthogonalMoEActor,
    batched_gram_schmidt,
    compute_orthogonality_metrics,
)
