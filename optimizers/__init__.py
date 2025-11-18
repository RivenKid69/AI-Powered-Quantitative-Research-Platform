"""
UPGD Optimizers for Continual Learning in Trading Bot

This module implements UPGD (Utility-based Perturbed Gradient Descent) optimizers
for mitigating catastrophic forgetting and loss of plasticity in continual learning.

Reference:
    Elsayed, M., & Mahmood, A. R. (2024). Addressing Loss of Plasticity and
    Catastrophic Forgetting in Continual Learning. In Proceedings of the 12th
    International Conference on Learning Representations (ICLR).
    https://openreview.net/forum?id=sKPzAXoylB

Available optimizers:
    - UPGD: Basic utility-based perturbed gradient descent
    - AdaptiveUPGD: UPGD with Adam-style adaptive learning rates
    - UPGDW: UPGD with decoupled weight decay (AdamW-style)
"""

from .upgd import UPGD
from .adaptive_upgd import AdaptiveUPGD
from .upgdw import UPGDW

__all__ = [
    "UPGD",
    "AdaptiveUPGD",
    "UPGDW",
]

__version__ = "1.0.0"
