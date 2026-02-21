# LunarLander-v3: Baseline vs Distillation (Teacher KL)

This repo runs a comparison experiment on **LunarLander-v3** for small student networks.

## Setup (conda)

```bash
conda env create -f environment.yml
conda activate lunarlander-distill
python -m pip install -e .
```

## Run the comparison experiment

Default: architectures `[ (4,), (8,), (16,), (64,) ]`, **200,000** steps each, **20** deterministic eval episodes, seed **123**.

```bash
python -m scripts.run_comparison \
  --hf-repo-id "sb3/ppo-LunarLander-v3" \
  --seed 123 \
  --total-timesteps 200000 \
  --n-eval-episodes 20
```

Outputs go to `outputs/<run_id>/`:
- `results.json`
- `comparison.png`

## Quick smoke run (fast)

```bash
python -m scripts.run_comparison \
  --hf-repo-id "sb3/ppo-LunarLander-v3" \
  --seed 123 \
  --total-timesteps 2000 \
  --n-eval-episodes 2
```

## Evaluate the teacher only (deterministic)

```bash
python -m scripts.eval_teacher \
  --hf-repo-id "sb3/ppo-LunarLander-v3" \
  --seed 123 \
  --n-eval-episodes 20
```

Outputs go to `outputs/teacher_eval_<timestamp>/`:
- `metrics.json`
- `returns.png`
- `lengths.png`

## Notes
- Distillation uses hybrid loss: `alpha * KL(teacher||student) + (1-alpha) * PPO_loss`, with `alpha=0.7`.
- Evaluation is deterministic: `model.predict(..., deterministic=True)` and per-episode seeds `seed + episode_index`.
- Teacher is downloaded from Hugging Face (SB3 `.zip` file) via `huggingface_hub`.
