from __future__ import annotations

import argparse
import sys
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

# Allow running this script without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from lunarlander_distill.distill_ppo import DistillConfig, DistillPPO
from lunarlander_distill.envs import maybe_set_torch_determinism, set_global_seeds
from lunarlander_distill.eval import evaluate_policy_deterministic
from lunarlander_distill.teacher import download_teacher_zip
from lunarlander_distill.tqdm_callback import TqdmConfig, TqdmProgressCallback


ARCHS = [(4,), (8,), (16,), (64,)]


@dataclass(frozen=True)
class RunConfig:
    env_id: str = "LunarLander-v3"
    archs: tuple[tuple[int, ...], ...] = tuple(ARCHS)
    total_timesteps: int = 200_000
    n_eval_episodes: int = 20
    seed: int = 123
    alpha: float = 0.7
    n_envs: int = 8
    device: str = "auto"
    hf_repo_id: str = "sb3/ppo-LunarLander-v3"
    hf_filename: str | None = None
    outputs_dir: Path = Path("outputs")


def _policy_kwargs_for_arch(arch: tuple[int, ...]) -> dict:
    # Separate pi/vf nets but same shape
    return {"net_arch": {"pi": list(arch), "vf": list(arch)}}


def _timestamp_run_id() -> str:
    return datetime.now().strftime("compare_%Y%m%d_%H%M%S")


def _save_plot(
    *,
    out_path: Path,
    sizes: list[int],
    baseline: list[float],
    distilled: list[float],
    env_id: str,
) -> None:
    x = list(range(len(sizes)))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], baseline, width=width, label="Baseline Reward")
    plt.bar([i + width / 2 for i in x], distilled, width=width, label="Distilled Reward")
    plt.xticks(x, [str(s) for s in sizes])
    plt.xlabel("Student hidden layer size")
    plt.ylabel("Mean eval reward (deterministic)")
    plt.title(f"{env_id}: Baseline PPO vs Teacher-Distilled PPO")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo-id", type=str, default="sb3/ppo-LunarLander-v3")
    ap.add_argument("--hf-filename", type=str, default=None)
    ap.add_argument("--env-id", type=str, default="LunarLander-v3")
    ap.add_argument("--total-timesteps", type=int, default=200_000)
    ap.add_argument("--n-eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    maybe_set_torch_determinism(True)
    set_global_seeds(int(args.seed))

    cfg = RunConfig(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed,
        alpha=args.alpha,
        n_envs=args.n_envs,
        device=args.device,
        hf_repo_id=args.hf_repo_id,
        hf_filename=args.hf_filename,
        outputs_dir=args.outputs_dir,
    )

    run_dir = cfg.outputs_dir / _timestamp_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher once
    teacher_zip = download_teacher_zip(repo_id=cfg.hf_repo_id, filename=cfg.hf_filename)
    teacher = PPO.load(teacher_zip, device=cfg.device)

    results = {
        "size": [a[0] for a in cfg.archs],
        "baseline_scores": [],
        "distilled_scores": [],
        "config": {
            **asdict(cfg),
            "outputs_dir": str(cfg.outputs_dir),
            "archs": [list(a) for a in cfg.archs],
        },
    }

    for arch in cfg.archs:
        size = arch[0]
        policy_kwargs = _policy_kwargs_for_arch(arch)

        # Baseline
        env = make_vec_env(
            cfg.env_id,
            n_envs=cfg.n_envs,
            seed=cfg.seed,
            vec_env_cls=DummyVecEnv,
        )
        baseline_model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            seed=cfg.seed,
            device=cfg.device,
            verbose=0,
        )
        baseline_cb = TqdmProgressCallback(
            TqdmConfig(
                total_timesteps=cfg.total_timesteps,
                desc=f"Baseline (arch={size})",
            )
        )
        baseline_model.learn(total_timesteps=cfg.total_timesteps, callback=baseline_cb)
        env.close()

        baseline_eval = evaluate_policy_deterministic(
            model=baseline_model,
            env_id=cfg.env_id,
            base_seed=cfg.seed,
            n_eval_episodes=cfg.n_eval_episodes,
        )
        results["baseline_scores"].append(baseline_eval["return_mean"])

        # Distilled
        env = make_vec_env(
            cfg.env_id,
            n_envs=cfg.n_envs,
            seed=cfg.seed,
            vec_env_cls=DummyVecEnv,
        )
        distill_model = DistillPPO(
            "MlpPolicy",
            env,
            teacher=teacher,
            distill=DistillConfig(alpha=cfg.alpha),
            policy_kwargs=policy_kwargs,
            seed=cfg.seed,
            device=cfg.device,
            verbose=0,
        )
        distill_cb = TqdmProgressCallback(
            TqdmConfig(
                total_timesteps=cfg.total_timesteps,
                desc=f"Distilled (arch={size})",
            )
        )
        distill_model.learn(total_timesteps=cfg.total_timesteps, callback=distill_cb)
        env.close()

        distill_eval = evaluate_policy_deterministic(
            model=distill_model,
            env_id=cfg.env_id,
            base_seed=cfg.seed,
            n_eval_episodes=cfg.n_eval_episodes,
        )
        results["distilled_scores"].append(distill_eval["return_mean"])

        print(
            f"arch={size:>3} | baseline={baseline_eval['return_mean']:.1f} | distilled={distill_eval['return_mean']:.1f}"
        )

    # Save results
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    # Plot
    plot_path = run_dir / "comparison.png"
    _save_plot(
        out_path=plot_path,
        sizes=results["size"],
        baseline=results["baseline_scores"],
        distilled=results["distilled_scores"],
        env_id=cfg.env_id,
    )

    print(f"Wrote: {results_path}")
    print(f"Wrote: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
