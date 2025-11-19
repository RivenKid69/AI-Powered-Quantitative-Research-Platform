"""
Adversarial Training Module for Robust Reinforcement Learning

This package implements adversarial training techniques for robust RL:
- State perturbation attacks (PGD, FGSM)
- State-Adversarial PPO (SA-PPO)
- Population-Based Training (PBT) scheduler

Based on research:
- "Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations" (NeurIPS 2020)
- "Population Based Training of Neural Networks" (DeepMind 2017)
"""

from adversarial.state_perturbation import (
    PerturbationConfig,
    StatePerturbation,
    test_loss_fn_policy,
    test_loss_fn_value,
)
from adversarial.sa_ppo import (
    SAPPOConfig,
    StateAdversarialPPO,
)
from adversarial.pbt_scheduler import (
    HyperparamConfig,
    PBTConfig,
    PopulationMember,
    PBTScheduler,
)

__all__ = [
    # State perturbation
    "PerturbationConfig",
    "StatePerturbation",
    "test_loss_fn_policy",
    "test_loss_fn_value",
    # SA-PPO
    "SAPPOConfig",
    "StateAdversarialPPO",
    # PBT
    "HyperparamConfig",
    "PBTConfig",
    "PopulationMember",
    "PBTScheduler",
]
