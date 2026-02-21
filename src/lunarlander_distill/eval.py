from __future__ import annotations

import gymnasium as gym
import numpy as np

from lunarlander_distill.envs import (
    ensure_lunarlander_v3_registered,
    maybe_set_torch_determinism,
    seed_everything_for_episode,
)


def evaluate_policy_deterministic(
    *,
    model,
    env_id: str,
    base_seed: int,
    n_eval_episodes: int,
    deterministic: bool = True,
) -> dict:
    """Policy evaluation for Gymnasium environments.

    - Uses `deterministic=...` in `model.predict`.
    - Episode seeds are `base_seed + episode_index`.
    """

    maybe_set_torch_determinism(True)

    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(n_eval_episodes):
        seed = seed_everything_for_episode(base_seed=base_seed, episode_index=ep)

        if env_id == "LunarLander-v3":
            ensure_lunarlander_v3_registered()

        try:
            env = gym.make(env_id)
        except Exception as e:
            # Provide a more actionable message for the common LunarLander version mismatch.
            name_not_found = getattr(getattr(gym, "error", None), "NameNotFound", None)
            version_not_found = getattr(getattr(gym, "error", None), "VersionNotFound", None)
            if (name_not_found is not None and isinstance(e, name_not_found)) or (
                version_not_found is not None and isinstance(e, version_not_found)
            ):
                available = [
                    k
                    for k in getattr(getattr(gym, "envs", None), "registry", {}).keys()
                    if isinstance(k, str) and k.startswith("LunarLander-")
                ]
                available_str = ", ".join(sorted(available)) if available else "<none>"
                raise RuntimeError(
                    f"Could not create env_id={env_id!r}. Available LunarLander envs: {available_str}. "
                    "(If you expected LunarLander-v3, install a Gymnasium version that provides it.)"
                ) from e
            raise
        try:
            # Best-effort: seed spaces too (some envs sample from these)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            obs, _ = env.reset(seed=seed)

            done = False
            ep_ret = 0.0
            ep_len = 0

            while not done:
                action, _ = model.predict(obs, deterministic=bool(deterministic))
                obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1

            returns.append(ep_ret)
            lengths.append(ep_len)
        finally:
            env.close()

    r = np.asarray(returns, dtype=np.float64)
    l = np.asarray(lengths, dtype=np.int64)

    return {
        "returns": returns,
        "lengths": lengths,
        "return_mean": float(r.mean()),
        "return_std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "length_mean": float(l.mean()),
        "length_std": float(l.std(ddof=1)) if len(l) > 1 else 0.0,
    }
