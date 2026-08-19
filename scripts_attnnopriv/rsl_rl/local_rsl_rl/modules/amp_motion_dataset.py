from __future__ import annotations

from collections.abc import Sequence
import glob
import json
from pathlib import Path

import torch


class AMPMotionDataset:
    """Mocap transition sampler for 30-D leg AMP observations."""

    AMP_OBS_DIM = 30

    def __init__(
        self,
        motion_files: str | Sequence[str],
        device: str | torch.device = "cpu",
        replay_buffer_size: int = 1_000_000,
        num_preload_transitions: int = 0,
    ):
        self.device = torch.device(device)
        self.motion_paths = self._resolve_motion_files(motion_files)
        self.motions = [self._load_motion(path) for path in self.motion_paths]
        if not self.motions:
            raise ValueError("AMPMotionDataset requires at least one motion file")
        self.motion_lengths = torch.tensor([motion.shape[0] for motion in self.motions], device=self.device)
        if torch.any(self.motion_lengths < 2):
            raise ValueError("Each AMP motion must contain at least two frames")

        self.replay_obs = None
        self.replay_next_obs = None
        self.replay_size = 0
        if replay_buffer_size > 0 and num_preload_transitions > 0:
            self._preload_replay_buffer(replay_buffer_size, num_preload_transitions)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.replay_obs is not None and self.replay_size > 0:
            indices = torch.randint(self.replay_size, (batch_size,), device=self.device)
            return self.replay_obs[indices], self.replay_next_obs[indices]

        motion_ids = torch.randint(len(self.motions), (batch_size,), device=self.device)
        obs = torch.empty(batch_size, self.AMP_OBS_DIM, device=self.device)
        next_obs = torch.empty_like(obs)
        for motion_id in torch.unique(motion_ids):
            mask = motion_ids == motion_id
            count = int(mask.sum().item())
            motion = self.motions[int(motion_id.item())]
            frame_ids = torch.randint(motion.shape[0], (count,), device=self.device)
            obs[mask] = motion[frame_ids]
            next_obs[mask] = motion[(frame_ids + 1) % motion.shape[0]]
        return obs, next_obs

    def _preload_replay_buffer(self, replay_buffer_size: int, num_preload_transitions: int):
        self.replay_size = min(int(replay_buffer_size), int(num_preload_transitions))
        if self.replay_size <= 0:
            return
        self.replay_obs = torch.empty(self.replay_size, self.AMP_OBS_DIM, device=self.device)
        self.replay_next_obs = torch.empty_like(self.replay_obs)

        chunk_size = 65_536
        start = 0
        while start < self.replay_size:
            end = min(start + chunk_size, self.replay_size)
            obs, next_obs = self._sample_direct(end - start)
            self.replay_obs[start:end] = obs
            self.replay_next_obs[start:end] = next_obs
            start = end

    def _sample_direct(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        replay_obs, replay_next_obs = self.replay_obs, self.replay_next_obs
        self.replay_obs, self.replay_next_obs = None, None
        try:
            return self.sample(batch_size)
        finally:
            self.replay_obs, self.replay_next_obs = replay_obs, replay_next_obs

    def _load_motion(self, path: Path) -> torch.Tensor:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        frames = torch.tensor(data["Frames"], dtype=torch.float32, device=self.device)
        if frames.dim() != 2 or frames.shape[1] < 49:
            raise ValueError(f"AMP motion {path} must have frames shaped [T, >=49], got {tuple(frames.shape)}")
        return torch.cat(
            [
                frames[:, 7:19],
                frames[:, 31:34],
                frames[:, 34:37],
                frames[:, 37:49],
            ],
            dim=-1,
        )

    def _resolve_motion_files(self, motion_files: str | Sequence[str]) -> list[Path]:
        if isinstance(motion_files, str):
            patterns = [motion_files]
        else:
            patterns = list(motion_files)
        paths: list[Path] = []
        for pattern in patterns:
            matches = sorted(glob.glob(pattern))
            if matches:
                paths.extend(Path(match) for match in matches)
            else:
                paths.append(Path(pattern))
        paths = [path for path in paths if path.is_file()]
        if not paths:
            raise FileNotFoundError(f"No AMP motion files matched: {patterns}")
        return paths
