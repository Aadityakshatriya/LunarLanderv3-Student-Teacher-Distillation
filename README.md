# LunarLander-v3 Distillation

Train a **heavy PPO teacher** on `LunarLander-v3`, then **distill** it into a **tiny PPO student** with KL divergence on action logits. The live dashboard shows **only**:
- `distill/loss` vs steps
- `distill/mean_reward` vs steps

**Setup**
```bash
conda env create -f environment.yml
conda activate lunarlander-distill
```

Progress bars and TensorBoard require extra packages:
```bash
python -m pip install rich tensorboard "setuptools==65.5.1"
```

**Train Teacher**
```bash
python train_teacher.py
```

Outputs:
- `./logs/best_model.zip` (teacher checkpoint)
- `./logs/tb/` (TensorBoard logs)

All evaluation/distillation utilities use a **local teacher path** only (no HF downloads). By default they read:
`./logs/best_model.zip` and can be overridden with `--teacher-path`.

**Local-Only Utilities**
```bash
python -m scripts.verify_setup --teacher-path "./logs/best_model.zip"
python -m scripts.eval_teacher --teacher-path "./logs/best_model.zip"
python -m scripts.run_comparison --teacher-path "./logs/best_model.zip"
```

**Distill Tiny Student (KL)**
```bash
python scripts/distill_student.py \
  --teacher-path "./logs/best_model.zip" \
  --total-timesteps 1000000 \
  --eval-every 10000 \
  --eval-episodes 10 \
  --student-hidden 16 \
  --log-dir "./logs/distill_tb" \
  --out-path "./distilled_student_lunar.zip"
```

**Comparison Table (Teacher vs Student)**
```bash
python scripts/compare_models.py \
  --teacher-path "./logs/best_model.zip" \
  --student-path "./distilled_student_lunar.zip" \
  --episodes 100 \
  --log-dir "./logs/distill_tb"
```

This prints a markdown table and logs it to TensorBoard (Text tab).

**Live Dashboard**
```bash
tensorboard --logdir "./logs/distill_tb"
```
Open the URL it prints (usually `http://localhost:6006`).

**Notes**
- `train_teacher.py` uses an eval callback and a custom tqdm progress bar with live metrics.
- `scripts/distill_student.py` logs only `distill/loss` and `distill/mean_reward`.
- `scripts/compare_models.py` estimates landing success rate from `info["is_success"]` or reward ≥ 200.
