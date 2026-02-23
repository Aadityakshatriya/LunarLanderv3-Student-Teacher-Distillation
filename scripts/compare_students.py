from __future__ import annotations

import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

# Allow running this script without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def count_params(model: PPO) -> int:
    return sum(p.numel() for p in model.policy.parameters())


def evaluate_with_success(
    model: PPO,
    env_id: str,
    n_episodes: int,
    success_threshold: float = 200.0,
) -> tuple[float, float, float]:
    env = gym.make(env_id)
    returns = []
    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = truncated = False
        ep_ret = 0.0
        last_info = {}

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += float(reward)
            last_info = info

        returns.append(ep_ret)
        if "is_success" in last_info:
            successes += int(bool(last_info["is_success"]))
        else:
            successes += int(ep_ret >= success_threshold)

    env.close()
    r = np.asarray(returns, dtype=np.float64)
    return float(r.mean()), float(r.std(ddof=1) if len(r) > 1 else 0.0), float(successes) / float(n_episodes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="LunarLander-v3")
    ap.add_argument("--baseline-path", type=str, default="./baseline_student_lunar.zip")
    ap.add_argument("--distilled-path", type=str, default="./distilled_student_lunar.zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--log-dir", type=str, default="./logs/distill_tb")
    args = ap.parse_args()

    baseline_path = Path(args.baseline_path)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {baseline_path}")
    distilled_path = Path(args.distilled_path)
    if not distilled_path.exists():
        raise FileNotFoundError(f"Distilled model not found: {distilled_path}")

    baseline = PPO.load(baseline_path, device="auto")
    distilled = PPO.load(distilled_path, device="auto")

    baseline_mean, _baseline_std, baseline_succ = evaluate_with_success(
        baseline, args.env_id, n_episodes=int(args.episodes)
    )
    distilled_mean, _distilled_std, distilled_succ = evaluate_with_success(
        distilled, args.env_id, n_episodes=int(args.episodes)
    )

    rows = [
        ("Student (Baseline)", count_params(baseline), baseline_mean, baseline_succ),
        ("Student (Distilled)", count_params(distilled), distilled_mean, distilled_succ),
    ]

    table = [
        "| Model | Parameters (approx) | Mean Reward (100 eps) | Landing Success Rate |",
        "|---|---:|---:|---:|",
    ]
    for name, params, mean_reward, succ in rows:
        table.append(f"| {name} | ~{params:,} | {mean_reward:.1f} | {succ*100:.1f}% |")

    md_table = "\n".join(table)
    print(md_table)

    writer = SummaryWriter(log_dir=str(Path(args.log_dir)))
    writer.add_text("comparison/table", md_table)
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
