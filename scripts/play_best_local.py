from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Allow running this script without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lunarlander_distill.envs import ensure_lunarlander_v3_registered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=Path("logs/best_model.zip"))
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Episode seed. If omitted, a random seed is sampled.",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic=True in model.predict().",
    )
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    seed = int(args.seed) if args.seed is not None else int(np.random.randint(0, 2**31 - 1))
    print(f"Using seed={seed}")

    ensure_lunarlander_v3_registered()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {args.model_path}. "
            "Run train_teacher.py first to generate logs/best_model.zip."
        )

    model = PPO.load(str(args.model_path), device=args.device)

    env = gym.make("LunarLander-v3", render_mode="human")
    try:
        obs, _info = env.reset(seed=seed)
        terminated = truncated = False
        ep_ret = 0.0
        ep_len = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1

        print(f"Episode return={ep_ret:.2f} length={ep_len}")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())