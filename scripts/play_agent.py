from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from lunarlander_distill.envs import set_global_seeds


def resolve_model_path(agent: str) -> Path:
    if agent == "teacher":
        return Path("./logs/best_model.zip")
    if agent == "baseline":
        return Path("./baseline_student_lunar.zip")
    if agent == "distilled":
        return Path("./distilled_student_lunar.zip")
    raise ValueError(f"Unknown agent: {agent}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="LunarLander-v3")
    ap.add_argument(
        "--agent",
        type=str,
        choices=["teacher", "baseline", "distilled"],
        default="distilled",
    )
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--render-mode", type=str, default="human")
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()

    set_global_seeds(int(args.seed))
    np.random.seed(int(args.seed))

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = resolve_model_path(args.agent)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    render_mode = args.render_mode if args.render_mode.lower() != "none" else None
    env = gym.make(args.env_id, render_mode=render_mode)
    model = PPO.load(model_path, device="auto")

    for ep in range(int(args.episodes)):
        obs, _info = env.reset(seed=int(args.seed) + ep)
        done = truncated = False
        ep_ret = 0.0
        ep_len = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            if render_mode is not None:
                env.render()

        print(f"Episode {ep + 1}: return={ep_ret:.2f}, length={ep_len}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
