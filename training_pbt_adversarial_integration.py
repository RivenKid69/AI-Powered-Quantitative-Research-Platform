"""
Integration module for Population-Based Training + Adversarial Training.

This module provides integration between PBT, SA-PPO, and the main training pipeline.
It wraps DistributionalPPO with adversarial training capabilities and coordinates
population-based hyperparameter optimization.

Usage:
    from training_pbt_adversarial_integration import (
        create_pbt_sappo_model,
        PBTTrainingCoordinator,
        load_pbt_adversarial_config,
    )

    # Load config
    config = load_pbt_adversarial_config("configs/config_pbt_adversarial.yaml")

    # Create coordinator
    coordinator = PBTTrainingCoordinator(config)

    # Run training
    coordinator.train()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from adversarial import (
    HyperparamConfig,
    PBTConfig,
    PBTScheduler,
    PopulationMember,
    SAPPOConfig,
    StateAdversarialPPO,
    PerturbationConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class PBTAdversarialConfig:
    """Combined configuration for PBT + Adversarial Training.

    Attributes:
        pbt_enabled: Whether PBT is enabled
        adversarial_enabled: Whether adversarial training is enabled
        pbt: PBT configuration
        adversarial: SA-PPO configuration
        base_config_path: Path to base training config (e.g., config_train.yaml)
    """
    pbt_enabled: bool = True
    adversarial_enabled: bool = True
    pbt: Optional[PBTConfig] = None
    adversarial: Optional[SAPPOConfig] = None
    base_config_path: Optional[str] = None


def load_pbt_adversarial_config(config_path: str) -> PBTAdversarialConfig:
    """Load PBT + Adversarial configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        PBTAdversarialConfig instance
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Parse PBT config
    pbt_enabled = data.get("pbt", {}).get("enabled", True)
    pbt_config = None
    if pbt_enabled:
        pbt_data = data["pbt"]

        # Parse hyperparameters
        hyperparams = []
        for hp_data in pbt_data.get("hyperparams", []):
            hp = HyperparamConfig(
                name=hp_data["name"],
                min_value=hp_data.get("min_value"),
                max_value=hp_data.get("max_value"),
                values=hp_data.get("values"),
                perturbation_factor=hp_data.get("perturbation_factor", 1.2),
                resample_probability=hp_data.get("resample_probability", 0.25),
                is_log_scale=hp_data.get("is_log_scale", False),
            )
            hyperparams.append(hp)

        pbt_config = PBTConfig(
            population_size=pbt_data.get("population_size", 10),
            perturbation_interval=pbt_data.get("perturbation_interval", 5),
            hyperparams=hyperparams,
            exploit_method=pbt_data.get("exploit_method", "truncation"),
            explore_method=pbt_data.get("explore_method", "both"),
            truncation_ratio=pbt_data.get("truncation_ratio", 0.2),
            checkpoint_dir=pbt_data.get("checkpoint_dir", "pbt_checkpoints"),
            metric_name=pbt_data.get("metric_name", "mean_reward"),
            metric_mode=pbt_data.get("metric_mode", "max"),
            ready_percentage=pbt_data.get("ready_percentage", 0.8),
        )

    # Parse Adversarial config
    # Only enable if adversarial key exists in YAML or use default
    adversarial_data = data.get("adversarial", {})
    adversarial_enabled = adversarial_data.get("enabled", True) if adversarial_data else False
    adversarial_config = None
    if adversarial_enabled and adversarial_data:
        adv_data = adversarial_data

        # Parse perturbation config
        pert_data = adv_data.get("perturbation", {})
        perturbation_config = PerturbationConfig(
            epsilon=pert_data.get("epsilon", 0.075),
            attack_steps=pert_data.get("attack_steps", 3),
            attack_lr=pert_data.get("attack_lr", 0.03),
            random_init=pert_data.get("random_init", True),
            norm_type=pert_data.get("norm_type", "linf"),
            clip_min=pert_data.get("clip_min"),
            clip_max=pert_data.get("clip_max"),
            attack_method=pert_data.get("attack_method", "pgd"),
        )

        adversarial_config = SAPPOConfig(
            enabled=True,
            perturbation=perturbation_config,
            adversarial_ratio=adv_data.get("adversarial_ratio", 0.5),
            robust_kl_coef=adv_data.get("robust_kl_coef", 0.1),
            warmup_updates=adv_data.get("warmup_updates", 10),
            attack_policy=adv_data.get("attack_policy", True),
            attack_value=adv_data.get("attack_value", True),
            adaptive_epsilon=adv_data.get("adaptive_epsilon", False),
            epsilon_schedule=adv_data.get("epsilon_schedule", "constant"),
            epsilon_final=adv_data.get("epsilon_final", 0.05),
        )

    return PBTAdversarialConfig(
        pbt_enabled=pbt_enabled,
        adversarial_enabled=adversarial_enabled,
        pbt=pbt_config,
        adversarial=adversarial_config,
        base_config_path=data.get("base_config_path"),
    )


def create_sappo_wrapper(
    model: Any,
    sa_ppo_config: SAPPOConfig,
) -> StateAdversarialPPO:
    """Create SA-PPO wrapper for a PPO model.

    Args:
        model: PPO model to wrap (e.g., DistributionalPPO)
        sa_ppo_config: SA-PPO configuration

    Returns:
        StateAdversarialPPO wrapper instance
    """
    sa_ppo = StateAdversarialPPO(sa_ppo_config, model)
    sa_ppo.on_training_start()

    logger.info(
        f"SA-PPO wrapper created: enabled={sa_ppo_config.enabled}, "
        f"adversarial_ratio={sa_ppo_config.adversarial_ratio}, "
        f"epsilon={sa_ppo_config.perturbation.epsilon}"
    )

    return sa_ppo


class PBTTrainingCoordinator:
    """Coordinator for PBT + Adversarial Training.

    This class manages the full training pipeline with PBT and adversarial training.
    It coordinates:
    - Population initialization
    - Parallel training of population members
    - Exploitation and exploration steps
    - Adversarial training for each member
    - Performance tracking and checkpointing
    """

    def __init__(
        self,
        config: PBTAdversarialConfig,
        seed: Optional[int] = None,
    ):
        """Initialize PBT training coordinator.

        Args:
            config: PBT + Adversarial configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed

        # Initialize PBT scheduler if enabled
        self.pbt_scheduler = None
        if config.pbt_enabled and config.pbt is not None:
            self.pbt_scheduler = PBTScheduler(config.pbt, seed=seed)
            logger.info("PBT scheduler initialized")

        # Storage for SA-PPO wrappers for each population member
        self.sa_ppo_wrappers: Dict[int, StateAdversarialPPO] = {}

        logger.info(
            f"PBT Training Coordinator initialized: "
            f"pbt_enabled={config.pbt_enabled}, "
            f"adversarial_enabled={config.adversarial_enabled}"
        )

    def initialize_population(
        self,
        initial_hyperparams: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PopulationMember]:
        """Initialize PBT population.

        Args:
            initial_hyperparams: Optional initial hyperparameter configurations

        Returns:
            List of initialized population members
        """
        if self.pbt_scheduler is None:
            raise RuntimeError("PBT scheduler not initialized")

        population = self.pbt_scheduler.initialize_population(initial_hyperparams)
        logger.info(f"Population initialized with {len(population)} members")

        return population

    def create_member_model(
        self,
        member: PopulationMember,
        base_model_factory: Any,
        **model_kwargs: Any,
    ) -> Tuple[Any, Optional[StateAdversarialPPO]]:
        """Create model for a population member with optional SA-PPO wrapper.

        Args:
            member: Population member
            base_model_factory: Factory function to create base model
            **model_kwargs: Additional keyword arguments for model creation

        Returns:
            Tuple of (model, sa_ppo_wrapper)
            sa_ppo_wrapper is None if adversarial training is disabled
        """
        # Update model kwargs with member's hyperparameters
        model_kwargs.update(member.hyperparams)

        # Create base model
        model = base_model_factory(**model_kwargs)

        # Wrap with SA-PPO if enabled
        sa_ppo_wrapper = None
        if self.config.adversarial_enabled and self.config.adversarial is not None:
            sa_ppo_wrapper = create_sappo_wrapper(model, self.config.adversarial)
            self.sa_ppo_wrappers[member.member_id] = sa_ppo_wrapper

        logger.info(
            f"Model created for member {member.member_id} with hyperparams: "
            f"{member.hyperparams}"
        )

        return model, sa_ppo_wrapper

    def on_member_update_start(self, member: PopulationMember) -> None:
        """Called at the start of each training update for a member.

        Args:
            member: Population member
        """
        # Notify SA-PPO wrapper
        if member.member_id in self.sa_ppo_wrappers:
            self.sa_ppo_wrappers[member.member_id].on_update_start()

    def on_member_update_end(
        self,
        member: PopulationMember,
        performance: float,
        step: int,
        model_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Called at the end of each training update for a member.

        This method:
        1. Updates performance in PBT scheduler
        2. Checks if exploitation and exploration should occur
        3. Returns new model state and hyperparameters if PBT step occurred

        Args:
            member: Population member
            performance: Performance metric value
            step: Current training step
            model_state_dict: Model state dict to save (optional)

        Returns:
            Tuple of (new_model_state_dict, new_hyperparams)
            new_model_state_dict is None if no PBT step occurred
        """
        # Update performance in PBT scheduler
        if self.pbt_scheduler is not None:
            self.pbt_scheduler.update_performance(
                member, performance, step, model_state_dict
            )

        # Check if should exploit and explore
        new_state_dict = None
        new_hyperparams = member.hyperparams

        if self.pbt_scheduler is not None and self.pbt_scheduler.should_exploit_and_explore(member):
            new_state_dict, new_hyperparams, checkpoint_format = self.pbt_scheduler.exploit_and_explore(
                member, model_state_dict
            )

        # Notify SA-PPO wrapper
        if member.member_id in self.sa_ppo_wrappers:
            self.sa_ppo_wrappers[member.member_id].on_update_end()

        return new_state_dict, new_hyperparams

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from PBT and SA-PPO.

        Returns:
            Dictionary with training statistics
        """
        stats = {}

        # PBT stats
        if self.pbt_scheduler is not None:
            stats.update(self.pbt_scheduler.get_stats())

        # SA-PPO stats (aggregate from all members)
        if self.sa_ppo_wrappers:
            sa_ppo_stats = {}
            for member_id, sa_ppo in self.sa_ppo_wrappers.items():
                member_stats = sa_ppo.get_stats()
                for key, value in member_stats.items():
                    if key not in sa_ppo_stats:
                        sa_ppo_stats[key] = []
                    sa_ppo_stats[key].append(value)

            # Average stats across members
            for key, values in sa_ppo_stats.items():
                if isinstance(values[0], (int, float)):
                    stats[f"avg_{key}"] = sum(values) / len(values)

        return stats


def is_pbt_adversarial_enabled_by_default() -> bool:
    """Check if PBT + Adversarial Training is enabled by default.

    This function checks the default configuration to determine if
    PBT and Adversarial Training are enabled by default in the system.

    Returns:
        True if both PBT and Adversarial are enabled by default
    """
    # Check default config file
    default_config_path = "configs/config_pbt_adversarial.yaml"
    if os.path.exists(default_config_path):
        try:
            config = load_pbt_adversarial_config(default_config_path)
            return config.pbt_enabled and config.adversarial_enabled
        except Exception as e:
            logger.warning(f"Failed to load default config: {e}")
            return False

    # Check class defaults
    default_config = PBTAdversarialConfig()
    return default_config.pbt_enabled and default_config.adversarial_enabled


# Default instance for convenience
DEFAULT_PBT_ADVERSARIAL_CONFIG = PBTAdversarialConfig(
    pbt_enabled=True,
    adversarial_enabled=True,
)


if __name__ == "__main__":
    # Example usage
    print("PBT + Adversarial Training Integration Module")
    print(f"PBT enabled by default: {DEFAULT_PBT_ADVERSARIAL_CONFIG.pbt_enabled}")
    print(f"Adversarial enabled by default: {DEFAULT_PBT_ADVERSARIAL_CONFIG.adversarial_enabled}")
    print(f"System default enabled: {is_pbt_adversarial_enabled_by_default()}")
