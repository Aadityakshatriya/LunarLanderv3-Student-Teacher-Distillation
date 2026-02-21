from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
try:
    from tqdm.rich import tqdm
except Exception:  # pragma: no cover - fallback if rich/tqdm not available
    from tqdm import tqdm


class LiveProgressCallback(BaseCallback):
    def __init__(self, update_interval: int = 5_000):
        super().__init__()
        self.update_interval = update_interval
        self.last_update = 0
        self.pbar: tqdm | None = None

    def _on_training_start(self) -> None:
        total = self.locals["total_timesteps"] - self.model.num_timesteps
        self.pbar = tqdm(total=total)

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(self.training_env.num_envs)
            if (self.num_timesteps - self.last_update) >= self.update_interval:
                self.last_update = self.num_timesteps
                metrics = self._collect_metrics()
                if metrics:
                    self.pbar.set_postfix(metrics, refresh=False)
        return True

    def _collect_metrics(self) -> dict:
        logs = getattr(self.logger, "name_to_value", {}) or {}

        def fmt(key: str) -> str | None:
            val = logs.get(key, None)
            if val is None:
                return None
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val)

        metrics: dict[str, str] = {}
        for key, label in [
            ("rollout/ep_rew_mean", "rew"),
            ("eval/mean_reward", "eval"),
            ("train/loss", "loss"),
            ("train/policy_loss", "pi"),
            ("train/value_loss", "vf"),
            ("train/entropy_loss", "ent"),
        ]:
            v = fmt(key)
            if v is not None:
                metrics[label] = v
        return metrics

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.refresh()
            self.pbar.close()


class EvalLossCallback(BaseCallback):
    """Attach latest training loss to eval logs so it appears in the eval UI cards."""

    def _on_step(self) -> bool:
        logs = getattr(self.logger, "name_to_value", {}) or {}
        loss = logs.get("train/loss")
        if loss is not None:
            try:
                loss_val = float(loss)
            except Exception:
                loss_val = None
            if loss_val is not None:
                self.logger.record("eval/loss", loss_val)
                # Flush right away so it appears alongside eval metrics at this timestep.
                # EvalCallback already dumped once; this adds a second dump at same step
                # with the additional eval/loss scalar.
                parent = getattr(self, "parent", None)
                if parent is not None:
                    self.logger.dump(getattr(parent, "num_timesteps", self.num_timesteps))
        return True


def main() -> None:
    env_id = "LunarLander-v3"
    n_envs = 8

    # Train envs (vectorized) + separate eval env
    env = make_vec_env(env_id, n_envs=n_envs)
    eval_env = gym.make(env_id)

    # Eval every 10k steps total -> 10k / n_envs per env
    eval_freq = max(1, 10_000 // n_envs)

    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=EvalLossCallback(),
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=1024,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(log_dir / "tb"),
        verbose=1,
    )

    callbacks = CallbackList([eval_callback, LiveProgressCallback()])

    model.learn(
        total_timesteps=1_000_000,
        callback=callbacks,
        progress_bar=False,
        log_interval=1,
    )

    best_model_path = log_dir / "best_model.zip"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Missing best model at: {best_model_path}")

    best_model = PPO.load(best_model_path)

    mean_reward, std_reward = evaluate_policy(
        best_model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )

    print(
        f"Final Evaluation (20 episodes) - Mean Reward: {mean_reward:.2f} \u00b1 {std_reward:.2f}"
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
