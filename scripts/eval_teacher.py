from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Allow running this script without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lunarlander_distill.envs import maybe_set_torch_determinism, set_global_seeds
from lunarlander_distill.eval import evaluate_policy_deterministic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--teacher-path",
        type=Path,
        default=Path("./logs/best_model.zip"),
        help="Path to locally trained teacher (from train_teacher.py).",
    )
    ap.add_argument("--env-id", type=str, default="LunarLander-v3")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-eval-episodes", type=int, default=20)
    ap.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (deterministic=False) like SB3 defaults.",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    maybe_set_torch_determinism(True)
    set_global_seeds(int(args.seed))

    run_dir = args.outputs_dir / datetime.now().strftime("teacher_eval_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    teacher_path = Path(args.teacher_path)
    if not teacher_path.exists():
        raise FileNotFoundError(
            f"Teacher not found at {teacher_path}. "
            "Train a teacher with train_teacher.py or pass --teacher-path."
        )
    teacher = PPO.load(teacher_path, device=args.device)

    metrics = evaluate_policy_deterministic(
        model=teacher,
        env_id=args.env_id,
        base_seed=int(args.seed),
        n_eval_episodes=int(args.n_eval_episodes),
        deterministic=not bool(args.stochastic),
    )

    out = {
        "config": {
            "teacher_path": str(teacher_path),
            "env_id": args.env_id,
            "seed": int(args.seed),
            "n_eval_episodes": int(args.n_eval_episodes),
            "stochastic": bool(args.stochastic),
            "device": args.device,
        },
        **metrics,
    }

    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True))

    plt.figure()
    plt.plot(out["returns"], marker="o")
    plt.title("Teacher returns (deterministic)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "returns.png")
    plt.close()

    plt.figure()
    plt.plot(out["lengths"], marker="o")
    plt.title("Teacher episode lengths (deterministic)")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "lengths.png")
    plt.close()

    print(f"Wrote: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
