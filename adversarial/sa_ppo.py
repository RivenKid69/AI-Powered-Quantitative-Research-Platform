"""
State-Adversarial PPO (SA-PPO) Implementation

Extends DistributionalPPO with adversarial training for robustness.

Based on "Robust Deep Reinforcement Learning against Adversarial Perturbations
on State Observations" (NeurIPS 2020 Spotlight).

Key features:
1. Adversarial state perturbations during training
2. Robust KL regularization based on SA-MDP
3. Mixed clean/adversarial training for balance
4. Compatible with existing DistributionalPPO architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from adversarial.state_perturbation import PerturbationConfig, StatePerturbation

logger = logging.getLogger(__name__)


@dataclass
class SAPPOConfig:
    """Configuration for State-Adversarial PPO.

    Attributes:
        enabled: Whether to enable adversarial training
        perturbation: Configuration for state perturbations
        adversarial_ratio: Ratio of adversarial samples (0.0 = all clean, 1.0 = all adversarial)
        robust_kl_coef: Coefficient for robust KL regularization
        warmup_updates: Number of updates before enabling adversarial training
        attack_policy: Whether to attack policy loss
        attack_value: Whether to attack value loss
        adaptive_epsilon: Whether to adapt epsilon based on training progress
        epsilon_schedule: Schedule for epsilon adaptation ('constant', 'linear', 'cosine')
        epsilon_final: Final epsilon value for schedule
    """
    enabled: bool = True
    perturbation: PerturbationConfig = None
    adversarial_ratio: float = 0.5
    robust_kl_coef: float = 0.1
    warmup_updates: int = 10
    attack_policy: bool = True
    attack_value: bool = True
    adaptive_epsilon: bool = False
    epsilon_schedule: str = "constant"
    epsilon_final: float = 0.05

    def __post_init__(self) -> None:
        """Initialize default perturbation config if not provided."""
        if self.perturbation is None:
            self.perturbation = PerturbationConfig()

        # Validate parameters
        if not 0.0 <= self.adversarial_ratio <= 1.0:
            raise ValueError(f"adversarial_ratio must be in [0, 1], got {self.adversarial_ratio}")
        if self.robust_kl_coef < 0:
            raise ValueError(f"robust_kl_coef must be >= 0, got {self.robust_kl_coef}")
        if self.warmup_updates < 0:
            raise ValueError(f"warmup_updates must be >= 0, got {self.warmup_updates}")
        if self.epsilon_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(f"epsilon_schedule must be 'constant', 'linear', or 'cosine', got {self.epsilon_schedule}")


class StateAdversarialPPO:
    """State-Adversarial PPO wrapper for robust training.

    This class wraps a PPO model and adds adversarial training capabilities.
    It can be used with DistributionalPPO or any PPO-based algorithm.
    """

    def __init__(
        self,
        config: SAPPOConfig,
        model: Any,  # DistributionalPPO or similar
    ):
        """Initialize SA-PPO wrapper.

        Args:
            config: SA-PPO configuration
            model: PPO model to wrap (e.g., DistributionalPPO)
        """
        self.config = config
        self.model = model
        self.perturbation_gen = StatePerturbation(config.perturbation)

        self._update_count = 0
        self._adversarial_enabled = False

        # Statistics
        self._total_adversarial_samples = 0
        self._total_clean_samples = 0
        self._total_robust_kl_penalty = 0.0

    @property
    def is_adversarial_enabled(self) -> bool:
        """Check if adversarial training is currently enabled."""
        return self._adversarial_enabled and self._update_count >= self.config.warmup_updates

    def reset_stats(self) -> None:
        """Reset training statistics."""
        self._total_adversarial_samples = 0
        self._total_clean_samples = 0
        self._total_robust_kl_penalty = 0.0
        self.perturbation_gen.reset_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary with SA-PPO statistics
        """
        total_samples = self._total_adversarial_samples + self._total_clean_samples
        perturbation_stats = self.perturbation_gen.get_stats()

        return {
            "sa_ppo/enabled": self.is_adversarial_enabled,
            "sa_ppo/update_count": self._update_count,
            "sa_ppo/adversarial_samples": self._total_adversarial_samples,
            "sa_ppo/clean_samples": self._total_clean_samples,
            "sa_ppo/adversarial_ratio": (
                self._total_adversarial_samples / total_samples if total_samples > 0 else 0.0
            ),
            "sa_ppo/robust_kl_penalty": (
                self._total_robust_kl_penalty / self._update_count if self._update_count > 0 else 0.0
            ),
            "sa_ppo/current_epsilon": self._get_current_epsilon(),
            "sa_ppo/attack_count": perturbation_stats.get("attack_count", 0),
            "sa_ppo/avg_perturbation_norm": perturbation_stats.get("avg_perturbation_norm", 0.0),
        }

    def on_training_start(self) -> None:
        """Called when training starts."""
        self._adversarial_enabled = self.config.enabled
        logger.info(
            f"SA-PPO initialized: enabled={self.config.enabled}, "
            f"warmup_updates={self.config.warmup_updates}, "
            f"adversarial_ratio={self.config.adversarial_ratio}"
        )

    def on_update_start(self) -> None:
        """Called at the start of each update."""
        self._update_count += 1

        # Update epsilon schedule
        if self.config.adaptive_epsilon:
            self._update_epsilon_schedule()

    def on_update_end(self) -> None:
        """Called at the end of each update."""
        pass

    def compute_adversarial_loss(
        self,
        states: Tensor,
        actions: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_log_probs: Tensor,
        old_values: Optional[Tensor] = None,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute adversarial loss with robust training.

        This method:
        1. Splits batch into clean and adversarial samples
        2. Generates adversarial perturbations for adversarial samples
        3. Computes loss on mixed batch
        4. Adds robust KL regularization
        5. Includes entropy regularization for exploration

        Args:
            states: State observations [batch_size, ...]
            actions: Actions taken [batch_size, ...]
            advantages: Advantage estimates [batch_size]
            returns: Target returns [batch_size]
            old_log_probs: Log probs from old policy [batch_size]
            old_values: Values from old policy [batch_size] (optional)
            clip_range: PPO clip range
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient

        Returns:
            Tuple of (total_loss, info_dict)
        """
        info = {}

        # Check if adversarial training is enabled
        if not self.is_adversarial_enabled:
            # Fall back to standard training
            return self._compute_standard_loss(
                states, actions, advantages, returns, old_log_probs, old_values, clip_range, ent_coef, vf_coef
            )

        batch_size = states.size(0)
        num_adversarial = int(batch_size * self.config.adversarial_ratio)
        num_clean = batch_size - num_adversarial

        # Update sample counts
        self._total_adversarial_samples += num_adversarial
        self._total_clean_samples += num_clean

        # Split into clean and adversarial samples
        # For simplicity, take first num_clean as clean, rest as adversarial
        states_clean = states[:num_clean]
        states_adv_base = states[num_clean:]

        actions_clean = actions[:num_clean]
        actions_adv = actions[num_clean:]

        advantages_clean = advantages[:num_clean]
        advantages_adv = advantages[num_clean:]

        returns_clean = returns[:num_clean]
        returns_adv = returns[num_clean:]

        old_log_probs_clean = old_log_probs[:num_clean]
        old_log_probs_adv = old_log_probs[num_clean:]

        # Generate adversarial perturbations for policy
        states_adv_perturbed = states_adv_base
        if self.config.attack_policy and num_adversarial > 0:
            # Create policy loss function for attack
            def policy_loss_fn(s_perturbed: Tensor) -> Tensor:
                dist = self.model.policy.get_distribution(s_perturbed)
                log_probs = dist.log_prob(actions_adv)
                ratio = torch.exp(log_probs - old_log_probs_adv)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                # Negative because we want to maximize loss (worst-case)
                return -torch.min(ratio * advantages_adv, clipped_ratio * advantages_adv).mean()

            # Generate perturbations
            delta = self.perturbation_gen.generate_perturbation(states_adv_base, policy_loss_fn)
            states_adv_perturbed = states_adv_base + delta

        # Combine clean and adversarial states
        states_combined = torch.cat([states_clean, states_adv_perturbed], dim=0)
        actions_combined = torch.cat([actions_clean, actions_adv], dim=0)
        advantages_combined = torch.cat([advantages_clean, advantages_adv], dim=0)
        returns_combined = torch.cat([returns_clean, returns_adv], dim=0)
        old_log_probs_combined = torch.cat([old_log_probs_clean, old_log_probs_adv], dim=0)

        # Compute policy loss
        dist = self.model.policy.get_distribution(states_combined)
        log_probs = dist.log_prob(actions_combined)
        ratio = torch.exp(log_probs - old_log_probs_combined)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.min(ratio * advantages_combined, clipped_ratio * advantages_combined).mean()

        # Compute entropy for exploration
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)
        entropy_loss = -torch.mean(entropy)

        # Compute value loss (optionally with adversarial perturbations)
        if self.config.attack_value and num_adversarial > 0:
            # Generate perturbations for value function
            def value_loss_fn(s_perturbed: Tensor) -> Tensor:
                values = self.model.policy.predict_values(s_perturbed)
                return nn.functional.mse_loss(values, returns_adv)

            delta_value = self.perturbation_gen.generate_perturbation(states_adv_base, value_loss_fn)
            states_adv_value = states_adv_base + delta_value

            # Combine for value computation
            states_value_combined = torch.cat([states_clean, states_adv_value], dim=0)
        else:
            states_value_combined = states_combined

        values = self.model.policy.predict_values(states_value_combined)
        value_loss = nn.functional.mse_loss(values, returns_combined)

        # Compute robust KL regularization
        robust_kl_penalty = 0.0
        if self.config.robust_kl_coef > 0 and num_adversarial > 0:
            # KL divergence between clean and adversarial policies
            with torch.no_grad():
                dist_clean = self.model.policy.get_distribution(states_adv_base)
                log_probs_clean = dist_clean.log_prob(actions_adv)

            dist_adv = self.model.policy.get_distribution(states_adv_perturbed)
            log_probs_adv = dist_adv.log_prob(actions_adv)

            # KL(clean || adversarial) - penalize large policy changes
            kl_div = (torch.exp(log_probs_clean) * (log_probs_clean - log_probs_adv)).mean()
            robust_kl_penalty = self.config.robust_kl_coef * kl_div
            self._total_robust_kl_penalty += robust_kl_penalty.item()

        # Total loss with entropy regularization (CRITICAL FIX)
        # Standard PPO loss includes entropy to encourage exploration and prevent policy collapse
        # References: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
        total_loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss + robust_kl_penalty

        # Info dict
        info.update({
            "sa_ppo/policy_loss": policy_loss.item(),
            "sa_ppo/value_loss": value_loss.item(),
            "sa_ppo/entropy_loss": entropy_loss.item(),
            "sa_ppo/entropy": -entropy_loss.item(),  # Actual entropy value (positive)
            "sa_ppo/robust_kl_penalty": robust_kl_penalty if isinstance(robust_kl_penalty, float) else robust_kl_penalty.item(),
            "sa_ppo/num_adversarial": num_adversarial,
            "sa_ppo/num_clean": num_clean,
        })

        return total_loss, info

    def _compute_standard_loss(
        self,
        states: Tensor,
        actions: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_log_probs: Tensor,
        old_values: Optional[Tensor],
        clip_range: float,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute standard PPO loss without adversarial training.

        Args:
            states: State observations [batch_size, ...]
            actions: Actions taken [batch_size, ...]
            advantages: Advantage estimates [batch_size]
            returns: Target returns [batch_size]
            old_log_probs: Log probs from old policy [batch_size]
            old_values: Values from old policy [batch_size] (optional)
            clip_range: PPO clip range
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient

        Returns:
            Tuple of (total_loss, info_dict)
        """
        # Standard PPO policy loss
        dist = self.model.policy.get_distribution(states)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Compute entropy for exploration (CRITICAL FIX)
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)
        entropy_loss = -torch.mean(entropy)

        # Standard value loss
        values = self.model.policy.predict_values(states)
        value_loss = nn.functional.mse_loss(values, returns)

        # Total loss with entropy regularization (CRITICAL FIX)
        # Standard PPO loss includes entropy to encourage exploration and prevent policy collapse
        # References: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
        total_loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

        info = {
            "sa_ppo/policy_loss": policy_loss.item(),
            "sa_ppo/value_loss": value_loss.item(),
            "sa_ppo/entropy_loss": entropy_loss.item(),
            "sa_ppo/entropy": -entropy_loss.item(),  # Actual entropy value (positive)
            "sa_ppo/robust_kl_penalty": 0.0,
        }

        return total_loss, info

    def _get_current_epsilon(self) -> float:
        """Get current epsilon value based on schedule."""
        if not self.config.adaptive_epsilon:
            return self.config.perturbation.epsilon

        # Assuming some max_updates for scheduling (can be made configurable)
        max_updates = 1000  # TODO: make this configurable
        progress = min(1.0, self._update_count / max_updates)

        epsilon_init = self.config.perturbation.epsilon
        epsilon_final = self.config.epsilon_final

        if self.config.epsilon_schedule == "linear":
            epsilon = epsilon_init + (epsilon_final - epsilon_init) * progress
        elif self.config.epsilon_schedule == "cosine":
            import math
            epsilon = epsilon_final + 0.5 * (epsilon_init - epsilon_final) * (1 + math.cos(math.pi * progress))
        else:  # constant
            epsilon = epsilon_init

        return epsilon

    def _update_epsilon_schedule(self) -> None:
        """Update epsilon value based on schedule."""
        current_epsilon = self._get_current_epsilon()
        self.config.perturbation.epsilon = current_epsilon

    def apply_adversarial_augmentation(
        self,
        states: Tensor,
        actions: Tensor,
        advantages: Tensor,
        old_log_probs: Tensor,
        clip_range: float,
    ) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        """Apply adversarial augmentation to a batch of states.

        This method generates adversarial perturbations and returns augmented states
        for use in downstream loss computation. It does NOT compute loss itself.

        Args:
            states: State observations [batch_size, ...]
            actions: Actions taken [batch_size, ...]
            advantages: Advantage estimates [batch_size]
            old_log_probs: Log probs from old policy [batch_size]
            clip_range: PPO clip range

        Returns:
            Tuple of (augmented_states, sample_mask, info_dict)
            - augmented_states: Combined clean + adversarial states [batch_size, ...]
            - sample_mask: Mask indicating which samples are adversarial [batch_size] (0=clean, 1=adv)
            - info_dict: Statistics dictionary
        """
        info = {}

        # Check if adversarial training is enabled
        if not self.is_adversarial_enabled:
            # Return original states with all-zero mask (all clean)
            sample_mask = torch.zeros(states.size(0), device=states.device, dtype=torch.float32)
            info.update({
                "sa_ppo/num_adversarial": 0,
                "sa_ppo/num_clean": states.size(0),
            })
            return states, sample_mask, info

        batch_size = states.size(0)
        num_adversarial = int(batch_size * self.config.adversarial_ratio)
        num_clean = batch_size - num_adversarial

        # Update sample counts
        self._total_adversarial_samples += num_adversarial
        self._total_clean_samples += num_clean

        # Split into clean and adversarial samples
        states_clean = states[:num_clean]
        states_adv_base = states[num_clean:]

        actions_adv = actions[num_clean:]
        advantages_adv = advantages[num_clean:]
        old_log_probs_adv = old_log_probs[num_clean:]

        # Generate adversarial perturbations for policy
        states_adv_perturbed = states_adv_base
        if self.config.attack_policy and num_adversarial > 0:
            # Create policy loss function for attack
            def policy_loss_fn(s_perturbed: Tensor) -> Tensor:
                dist = self.model.policy.get_distribution(s_perturbed)
                log_probs = dist.log_prob(actions_adv)
                ratio = torch.exp(log_probs - old_log_probs_adv)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                # Negative because we want to maximize loss (worst-case)
                return -torch.min(ratio * advantages_adv, clipped_ratio * advantages_adv).mean()

            # Generate perturbations
            delta = self.perturbation_gen.generate_perturbation(states_adv_base, policy_loss_fn)
            states_adv_perturbed = states_adv_base + delta

        # Combine clean and adversarial states
        states_combined = torch.cat([states_clean, states_adv_perturbed], dim=0)

        # Create sample mask (0 = clean, 1 = adversarial)
        sample_mask = torch.cat([
            torch.zeros(num_clean, device=states.device, dtype=torch.float32),
            torch.ones(num_adversarial, device=states.device, dtype=torch.float32),
        ], dim=0)

        # Info dict
        info.update({
            "sa_ppo/num_adversarial": num_adversarial,
            "sa_ppo/num_clean": num_clean,
        })

        return states_combined, sample_mask, info

    def compute_robust_kl_penalty(
        self,
        states_clean: Tensor,
        states_adv: Tensor,
        actions: Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute robust KL regularization between clean and adversarial policies.

        Args:
            states_clean: Clean state observations [batch_size, ...]
            states_adv: Adversarial (perturbed) state observations [batch_size, ...]
            actions: Actions to evaluate [batch_size, ...]

        Returns:
            Tuple of (robust_kl_penalty, info_dict)
        """
        info = {}

        if not self.is_adversarial_enabled or self.config.robust_kl_coef <= 0:
            return 0.0, info

        batch_size = states_clean.size(0)
        if batch_size == 0:
            return 0.0, info

        # KL divergence between clean and adversarial policies
        with torch.no_grad():
            dist_clean = self.model.policy.get_distribution(states_clean)
            log_probs_clean = dist_clean.log_prob(actions)

        dist_adv = self.model.policy.get_distribution(states_adv)
        log_probs_adv = dist_adv.log_prob(actions)

        # KL(clean || adversarial) - penalize large policy changes
        kl_div = (torch.exp(log_probs_clean) * (log_probs_clean - log_probs_adv)).mean()
        robust_kl_penalty = float((self.config.robust_kl_coef * kl_div).item())
        self._total_robust_kl_penalty += robust_kl_penalty

        info.update({
            "sa_ppo/robust_kl_penalty": robust_kl_penalty,
        })

        return robust_kl_penalty, info
