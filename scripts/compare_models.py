from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter


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
    ap.add_argument("--teacher-path", type=str, default="./logs/best_model.zip")
    ap.add_argument("--student-path", type=str, default="./distilled_student_lunar.zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--student-hidden", type=int, default=16)
    ap.add_argument("--log-dir", type=str, default="./logs/distill_tb")
    args = ap.parse_args()

    env_id = args.env_id

    teacher = PPO.load(args.teacher_path, device="auto")

    # Baseline (untrained tiny student)
    dummy_env = gym.make(env_id)
    baseline = PPO(
        "MlpPolicy",
        dummy_env,
        policy_kwargs={"net_arch": [int(args.student_hidden)]},
        verbose=0,
        device="auto",
    )

    distilled = PPO.load(args.student_path, device="auto")

    teacher_mean, _teacher_std, teacher_succ = evaluate_with_success(
        teacher, env_id, n_episodes=int(args.episodes)
    )
    baseline_mean, _baseline_std, baseline_succ = evaluate_with_success(
        baseline, env_id, n_episodes=int(args.episodes)
    )
    distilled_mean, _distilled_std, distilled_succ = evaluate_with_success(
        distilled, env_id, n_episodes=int(args.episodes)
    )

    rows = [
        ("Teacher (Full)", count_params(teacher), teacher_mean, teacher_succ),
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

    dummy_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
