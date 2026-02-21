from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

# Allow running this script without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lunarlander_distill.envs import ensure_lunarlander_v3_registered
from lunarlander_distill.teacher import download_teacher_zip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo-id", type=str, default="AminVilan/ppo-LunarLander-v3")
    ap.add_argument("--hf-filename", type=str, default="v01-ppo-LunarLanderV3.zip")
    ap.add_argument("--hf-token", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    # Gymnasium + Box2D sanity
    import gymnasium.envs.box2d  # noqa: F401
    import Box2D  # noqa: F401

    ensure_lunarlander_v3_registered()

    teacher_zip = download_teacher_zip(
        repo_id=args.hf_repo_id,
        filename=args.hf_filename,
        token=args.hf_token,
    )
    model = PPO.load(teacher_zip, device=args.device)

    env = gym.make("LunarLander-v3")
    obs, _info = env.reset(seed=123)

    terminated = truncated = False
    for _ in range(int(args.steps)):
        if terminated or truncated:
            break
        action, _ = model.predict(obs)  # nondeterministic by default
        obs, _reward, terminated, truncated, _info = env.step(action)

    env.close()
    print("OK: Gymnasium Box2D + LunarLander-v3 + teacher load + rollout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())