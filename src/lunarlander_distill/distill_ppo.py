from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch as th
import torch.nn.functional as F
from torch.distributions import kl_divergence

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


@dataclass(frozen=True)
class DistillConfig:
    alpha: float = 0.7


class DistillPPO(PPO):
    """PPO with a teacher KL auxiliary loss.

    Uses a hybrid loss:
        total_loss = (1 - alpha) * ppo_loss + alpha * KL(teacher || student)

    Designed for discrete-action teacher/student (LunarLander-v3).
    """

    def __init__(
        self,
        *args,
        teacher: PPO,
        distill: DistillConfig = DistillConfig(),
        **kwargs,
    ):
        if not (0.0 <= distill.alpha <= 1.0):
            raise ValueError("distill.alpha must be in [0, 1]")
        self.teacher = teacher
        self.distill = distill
        super().__init__(*args, **kwargs)

        # Keep teacher fixed
        self.teacher.policy.set_training_mode(False)
        for p in self.teacher.policy.parameters():
            p.requires_grad_(False)

    def _kl_teacher_student(self, obs: th.Tensor) -> th.Tensor:
        """Returns mean KL(teacher || student) for a minibatch."""

        # student distribution with gradients
        student_dist_obj = self.policy.get_distribution(obs)
        student_dist = getattr(student_dist_obj, "distribution", None)
        if student_dist is None:
            raise RuntimeError("Could not access student torch distribution")

        with th.no_grad():
            teacher_dist_obj = self.teacher.policy.get_distribution(obs)
            teacher_dist = getattr(teacher_dist_obj, "distribution", None)
            if teacher_dist is None:
                raise RuntimeError("Could not access teacher torch distribution")

        # torch.distributions.kl_divergence returns per-sample KL
        kl = kl_divergence(teacher_dist, student_dist)
        return kl.mean()

    def train(self) -> None:  # noqa: C901
        """Override PPO.train() to add a KL distillation term."""

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None
            else None
        )

        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        approx_kls = []
        kl_losses = []

        # train for n_epochs epochs
        continue_training = True
        for epoch in range(self.n_epochs):
            if not continue_training:
                break

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if self.action_space.__class__.__name__ == "Discrete":
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy
                log_ratio = log_prob - rollout_data.old_log_prob
                ratio = th.exp(log_ratio)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2))

                # Value function loss
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                ppo_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                kl_loss = self._kl_teacher_student(rollout_data.observations)
                total_loss = (1.0 - self.distill.alpha) * ppo_loss + self.distill.alpha * kl_loss

                # metrics
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_losses.append(float(kl_loss.item()))

                with th.no_grad():
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    approx_kl = th.mean((ratio - 1) - log_ratio).item()

                clip_fractions.append(clip_fraction)
                approx_kls.append(approx_kl)

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                total_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Log (SB3 logger)
        self.logger.record("train/entropy_loss", float(th.tensor(entropy_losses).mean()) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", float(th.tensor(pg_losses).mean()) if pg_losses else 0.0)
        self.logger.record("train/value_loss", float(th.tensor(value_losses).mean()) if value_losses else 0.0)
        self.logger.record("train/approx_kl", float(th.tensor(approx_kls).mean()) if approx_kls else 0.0)
        self.logger.record("train/clip_fraction", float(th.tensor(clip_fractions).mean()) if clip_fractions else 0.0)
        self.logger.record("train/explained_variance", float(explained_var))
        self.logger.record("train/distill_kl", float(th.tensor(kl_losses).mean()) if kl_losses else 0.0)
        self.logger.record("train/distill_alpha", float(self.distill.alpha))
        if hasattr(self, "_n_updates"):
            self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/clip_range", float(clip_range))
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", float(clip_range_vf))
