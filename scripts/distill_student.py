from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter

from lunarlander_distill.distill_ppo import DistillConfig, DistillPPO

class DistillMetricsCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env: gym.Env,
        writer: SummaryWriter,
        eval_every: int,
        eval_episodes: int,
        log_every: int,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.writer = writer
        self.eval_every = int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.log_every = int(log_every)
        self.last_eval = 0
        self.last_log = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_log) >= self.log_every:
            self.last_log = self.num_timesteps
            loss = None
            # Prefer distillation KL if exposed by DistillPPO
            if hasattr(self.model, "last_distill_kl") and self.model.last_distill_kl is not None:
                loss = float(self.model.last_distill_kl)
            else:
                logs = getattr(self.logger, "name_to_value", {}) or {}
                if "train/distill_kl" in logs:
                    loss = float(logs["train/distill_kl"])
                elif "train/loss" in logs:
                    loss = float(logs["train/loss"])
            if loss is not None:
                self.writer.add_scalar("distill/loss", loss, self.num_timesteps)

        if (self.num_timesteps - self.last_eval) >= self.eval_every:
            self.last_eval = self.num_timesteps
            mean_reward, _std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
            )
            self.writer.add_scalar("distill/mean_reward", mean_reward, self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.writer.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-path", type=str, default="./logs/best_model.zip")
    ap.add_argument("--env-id", type=str, default="LunarLander-v3")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=1_000)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--student-hidden", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
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

    # Tiny student with PPO + KL distillation
    student = DistillPPO(
        "MlpPolicy",
        env,
        teacher=teacher,
        distill=DistillConfig(
            alpha=float(args.alpha),
            temperature=float(args.temperature),
        ),
        policy_kwargs={"net_arch": [int(args.student_hidden)]},
        learning_rate=float(args.learning_rate),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        verbose=0,
        device="auto",
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    callback = DistillMetricsCallback(
        eval_env=eval_env,
        writer=writer,
        eval_every=int(args.eval_every),
        eval_episodes=int(args.eval_episodes),
        log_every=int(args.log_every),
    )

    student.learn(
        total_timesteps=int(args.total_timesteps),
        callback=callback,
        log_interval=1,
        progress_bar=True,
    )

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
