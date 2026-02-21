from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter


class BaselineMetricsCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env: gym.Env,
        writer: SummaryWriter,
        eval_every: int,
        eval_episodes: int,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.writer = writer
        self.eval_every = int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.last_eval = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_eval) >= self.eval_every:
            self.last_eval = self.num_timesteps
            mean_reward, _std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
            )
            self.writer.add_scalar("baseline/mean_reward", mean_reward, self.num_timesteps)

            logs = getattr(self.logger, "name_to_value", {}) or {}
            loss = logs.get("train/loss")
            if loss is not None:
                try:
                    self.writer.add_scalar("baseline/loss", float(loss), self.num_timesteps)
                except Exception:
                    pass
        return True

    def _on_training_end(self) -> None:
        self.writer.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="LunarLander-v3")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--student-hidden", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--log-dir", type=str, default="./logs/baseline_tb")
    ap.add_argument("--out-path", type=str, default="baseline_student_lunar.zip")
    args = ap.parse_args()

    set_random_seed(int(args.seed))

    env = make_vec_env(args.env_id, n_envs=int(args.n_envs), seed=int(args.seed))
    eval_env = gym.make(args.env_id)

    student = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [int(args.student_hidden)]},
        learning_rate=float(args.learning_rate),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        verbose=0,
        device=args.device,
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    callback = BaselineMetricsCallback(
        eval_env=eval_env,
        writer=writer,
        eval_every=int(args.eval_every),
        eval_episodes=int(args.eval_episodes),
    )

    student.learn(
        total_timesteps=int(args.total_timesteps),
        callback=callback,
        log_interval=1,
        progress_bar=True,
    )

    out_path = Path(args.out_path)
    student.save(out_path)
    print(f"Saved baseline student: {out_path}")

    mean_reward, std_reward = evaluate_policy(
        student,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )
    print(f"Final Baseline Eval (20 eps) - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

    writer.close()
    env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
