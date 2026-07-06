from __future__ import annotations

import torch
import torch.nn as nn


class PerTaskPopArt(nn.Module):
    """Per-task POPArt return normalization.

    The module tracks running return statistics independently for each task.
    Values passed to ``update`` are expected to be raw-scale returns.
    """

    def __init__(
        self,
        num_tasks: int,
        beta: float = 0.99999,
        eps: float = 1e-5,
        min_std: float = 1e-2,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        if num_tasks < 1:
            raise ValueError("num_tasks must be positive")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")

        self.num_tasks = num_tasks
        self.beta = beta
        self.eps = eps
        self.min_std = min_std

        mean = torch.full((num_tasks,), float(init_mean), device=device)
        second_moment = torch.full((num_tasks,), float(init_std) ** 2 + float(init_mean) ** 2, device=device)
        std = torch.full((num_tasks,), float(init_std), device=device)

        self.register_buffer("mean", mean)
        self.register_buffer("second_moment", second_moment)
        self.register_buffer("std", std)

    @torch.no_grad()
    def update(self, task_ids: torch.Tensor, returns: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        task_ids = self._check_task_ids(task_ids)
        returns = returns.view(-1).to(self.mean.device)
        if task_ids.shape[0] != returns.shape[0]:
            raise ValueError(f"task_ids and returns must have same batch size, got {task_ids.shape[0]} and {returns.shape[0]}")

        old_stats = self.get_stats()
        for task in torch.unique(task_ids):
            task_idx = int(task.item())
            mask = task_ids == task_idx
            task_returns = returns[mask]
            batch_mean = task_returns.mean()
            batch_second_moment = (task_returns * task_returns).mean()

            self.mean[task_idx] = self.beta * self.mean[task_idx] + (1.0 - self.beta) * batch_mean
            self.second_moment[task_idx] = (
                self.beta * self.second_moment[task_idx] + (1.0 - self.beta) * batch_second_moment
            )
            variance = torch.clamp(self.second_moment[task_idx] - self.mean[task_idx].pow(2), min=self.min_std**2)
            self.std[task_idx] = torch.sqrt(variance + self.eps)

        return old_stats, self.get_stats()

    def normalize(self, task_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        original_shape = values.shape
        task_ids = self._check_task_ids(task_ids)
        values_flat = values.view(-1).to(self.mean.device)
        mean, std = self._stats_for(task_ids)
        return ((values_flat - mean) / (std + self.eps)).view(original_shape)

    def denormalize(self, task_ids: torch.Tensor, values_norm: torch.Tensor) -> torch.Tensor:
        original_shape = values_norm.shape
        task_ids = self._check_task_ids(task_ids)
        values_flat = values_norm.view(-1).to(self.mean.device)
        mean, std = self._stats_for(task_ids)
        return (values_flat * (std + self.eps) + mean).view(original_shape)

    def get_stats(self, task_id: int | None = None) -> dict[str, torch.Tensor]:
        if task_id is None:
            return {
                "mean": self.mean.detach().clone(),
                "std": self.std.detach().clone(),
                "second_moment": self.second_moment.detach().clone(),
            }
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"task_id must be in [0, {self.num_tasks - 1}]")
        return {
            "mean": self.mean[task_id].detach().clone(),
            "std": self.std[task_id].detach().clone(),
            "second_moment": self.second_moment[task_id].detach().clone(),
        }

    def _stats_for(self, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if task_ids.dim() != 1:
            task_ids = task_ids.view(-1)
        return self.mean[task_ids], self.std[task_ids]

    def _check_task_ids(self, task_ids: torch.Tensor) -> torch.Tensor:
        task_ids = task_ids.long().view(-1).to(self.mean.device)
        if torch.any(task_ids < 0) or torch.any(task_ids >= self.num_tasks):
            raise ValueError(f"task_ids must be in [0, {self.num_tasks - 1}]")
        return task_ids
