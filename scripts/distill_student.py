from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter


def distill_step(
    *,
    teacher: PPO,
    student: PPO,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    temperature: float,
) -> float:
    with torch.no_grad():
        teacher_dist = teacher.policy.get_distribution(observations)
        teacher_logits = teacher_dist.distribution.logits / temperature

    student_dist = student.policy.get_distribution(observations)
    student_logits = student_dist.distribution.logits / temperature

    loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-path", type=str, default="./logs/best_model.zip")
    ap.add_argument("--env-id", type=str, default="LunarLander-v3")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--student-hidden", type=int, default=16)
    ap.add_argument("--log-dir", type=str, default="./logs/distill_tb")
    ap.add_argument("--out-path", type=str, default="distilled_student_lunar.zip")
    args = ap.parse_args()

    teacher_path = Path(args.teacher_path)
    if not teacher_path.exists():
        raise FileNotFoundError(
            f"Teacher not found at {teacher_path}. "
            "Train a teacher with train_teacher.py or pass --teacher-path."
        )

    env = make_vec_env(args.env_id, n_envs=int(args.n_envs))
    eval_env = gym.make(args.env_id)

    # Load teacher (frozen)
    teacher = PPO.load(teacher_path, device="auto")
    teacher.policy.set_training_mode(False)

    # Tiny student
    student = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [int(args.student_hidden)]},
        verbose=0,
        device="auto",
    )
    student.policy.set_training_mode(True)

    optimizer = torch.optim.Adam(student.policy.parameters(), lr=3e-4)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    obs = env.reset()
    num_steps = 0

    while num_steps < int(args.total_timesteps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=student.device)
        loss = distill_step(
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            observations=obs_t,
            temperature=float(args.temperature),
        )

        # Advance the env using the teacher to keep expert-state coverage
        actions, _ = teacher.predict(obs, deterministic=True)
        obs, _rewards, _dones, _infos = env.step(actions)

        num_steps += int(args.n_envs)

        writer.add_scalar("distill/loss", loss, num_steps)

        if num_steps % int(args.eval_every) == 0:
            mean_reward, _std_reward = evaluate_policy(
                student,
                eval_env,
                n_eval_episodes=int(args.eval_episodes),
                deterministic=True,
            )
            writer.add_scalar("distill/mean_reward", mean_reward, num_steps)

    out_path = Path(args.out_path)
    student.save(out_path)
    print(f"Saved distilled student: {out_path}")

    mean_reward, std_reward = evaluate_policy(
        student,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )
    print(f"Final Student Eval (20 eps) - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

    writer.close()
    env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
