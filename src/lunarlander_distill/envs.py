from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_seeds(base_seed: int, n_episodes: int) -> list[int]:
    return [base_seed + i for i in range(n_episodes)]


def seed_everything_for_episode(*, base_seed: int, episode_index: int) -> int:
    s = base_seed + episode_index
    set_global_seeds(s)
    return s


def maybe_set_torch_determinism(enable: bool = True) -> None:
    """Best-effort determinism for torch.

    Note: complete determinism across platforms/GPUs is not guaranteed.
    """

    if not enable:
        return

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_lunarlander_v3_registered() -> None:
    """Ensure `LunarLander-v3` is available.

    This intentionally does NOT alias `LunarLander-v2` to `LunarLander-v3`.
    If your Gymnasium install only provides `LunarLander-v2`, you must install
    a version that actually registers `LunarLander-v3`.
    """

    import gymnasium as gym

    registry = getattr(getattr(gym, "envs", None), "registry", None)
    if registry is None:
        raise RuntimeError("Gymnasium registry not found; cannot validate LunarLander-v3")

    if "LunarLander-v3" in registry:
        return

    available = [
        k for k in registry.keys() if isinstance(k, str) and k.startswith("LunarLander-")
    ]
    available_str = ", ".join(sorted(available)) if available else "<none>"
    raise RuntimeError(
        "LunarLander-v3 is not registered in this Gymnasium install. "
        f"Available LunarLander envs: {available_str}."
    )
