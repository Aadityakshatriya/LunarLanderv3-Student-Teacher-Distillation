from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TqdmConfig:
    total_timesteps: int
    desc: str


class TqdmProgressCallback(BaseCallback):
    """A simple tqdm progress bar for SB3 `learn()`.

    Updates by the delta of `self.num_timesteps` on each callback step.
    """

    def __init__(self, cfg: TqdmConfig):
        super().__init__()
        self.cfg = cfg
        self._pbar: Optional[tqdm] = None
        self._last_timesteps: int = 0

    def _on_training_start(self) -> None:
        self._last_timesteps = int(self.num_timesteps)
        self._pbar = tqdm(total=self.cfg.total_timesteps, desc=self.cfg.desc, unit="ts")

    def _on_step(self) -> bool:
        if self._pbar is None:
            return True
        now = int(self.num_timesteps)
        delta = max(0, now - self._last_timesteps)
        if delta:
            self._pbar.update(delta)
            self._last_timesteps = now
        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
