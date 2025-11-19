"""
Basic UPGD Optimizer

Implements the core Utility-based Perturbed Gradient Descent algorithm.

The algorithm protects useful weights from forgetting by applying smaller
modifications to high-utility parameters and larger perturbations to
low-utility parameters to maintain plasticity.

Algorithm:
    1. Compute utility for each parameter: u = -grad * param
    2. Track exponential moving average of utility with bias correction
    3. Find global maximum utility across all parameters
    4. Scale utility using sigmoid(utility / global_max)
    5. Apply gradient update with perturbation: param -= lr * (grad + noise) * (1 - scaled_utility)

Hyperparameters:
    - lr: Learning rate (default: 1e-5)
    - weight_decay: L2 penalty coefficient (default: 0.001)
    - beta_utility: EMA decay rate for utility (default: 0.999)
    - sigma: Gaussian noise standard deviation (default: 0.001)
"""

import torch
from typing import Any, Dict, Iterable, Optional, Union


class UPGD(torch.optim.Optimizer):
    """
    Utility-based Perturbed Gradient Descent optimizer.

    This optimizer protects useful weights from catastrophic forgetting while
    maintaining plasticity through controlled perturbations.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-5)
        weight_decay: L2 penalty coefficient (default: 0.001)
        beta_utility: Exponential moving average decay for utility (default: 0.999)
        sigma: Standard deviation of Gaussian noise for perturbation (default: 0.001)

    Example:
        >>> optimizer = UPGD(model.parameters(), lr=1e-4)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-5,
        weight_decay: float = 0.001,
        beta_utility: float = 0.999,
        sigma: float = 0.001,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta_utility < 1.0:
            raise ValueError(f"Invalid beta_utility parameter: {beta_utility}")
        if not 0.0 <= sigma:
            raise ValueError(f"Invalid sigma value: {sigma}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
        )
        super(UPGD, self).__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(UPGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # First pass: compute utilities and find global maximum
        global_max_util = torch.tensor(-torch.inf, device="cpu")

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Update exponential moving average of utility
                # Utility = -grad * param (measures importance of weight)
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )

                # Track global maximum utility
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max.cpu()

        # Move global_max_util back to appropriate device if needed
        # (will be used in device-specific operations below)

        # Second pass: apply updates with scaled utility
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                device = p.device

                # Bias correction for EMA
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]

                # Generate Gaussian noise for perturbation
                noise = torch.randn_like(p.grad) * group["sigma"]

                # Scale utility: sigmoid maps to [0, 1], high utility → close to 1
                # BUGFIX Bug #5: Add epsilon to prevent division by zero when global_max_util = 0
                # This can happen when all parameters are zero or all utilities are negative
                global_max_on_device = global_max_util.to(device)
                epsilon = 1e-8  # Small value to prevent division by zero
                scaled_utility = torch.sigmoid(
                    (state["avg_utility"] / bias_correction_utility) / (global_max_on_device + epsilon)
                )

                # Apply weight decay (L2 regularization)
                # Update: param -= lr * (grad + noise) * (1 - scaled_utility) - lr * weight_decay * param
                # Rearranged: param *= (1 - lr * weight_decay); param -= 2*lr * (grad + noise) * (1 - scaled_utility)
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1 - scaled_utility),
                    alpha=-2.0 * group["lr"],
                )

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Resets the gradients of all optimized parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zero.
                        This can slightly improve performance.
        """
        super().zero_grad(set_to_none=set_to_none)
