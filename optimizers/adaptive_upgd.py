"""
Adaptive UPGD Optimizer

Combines UPGD with Adam-style adaptive learning rates for improved performance
in deep neural networks with heterogeneous parameter scales.

This optimizer combines:
1. Utility-based weight protection from UPGD
2. Adaptive learning rates from Adam (first and second moments)
3. Bias correction for both utility and moment estimates

Algorithm:
    1. Compute utility: u = -grad * param
    2. Update first moment: m = beta1 * m + (1-beta1) * grad
    3. Update second moment: v = beta2 * v + (1-beta2) * grad^2
    4. Apply bias correction to all three
    5. Scale utility using sigmoid(utility / global_max)
    6. Update: param -= lr * (m / (sqrt(v) + eps) + noise) * (1 - scaled_utility)

This is particularly effective for RL tasks with PPO where parameter scales vary.
"""

import torch
from typing import Any, Dict, Iterable, Optional, Union


class AdaptiveUPGD(torch.optim.Optimizer):
    """
    Adaptive UPGD optimizer combining utility-based protection with Adam.

    This optimizer provides adaptive learning rates while protecting important
    weights from catastrophic forgetting.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-5)
        weight_decay: L2 penalty coefficient (default: 0.001)
        beta_utility: EMA decay for utility (default: 0.999)
        sigma: Standard deviation of noise (default: 0.001)
        beta1: Adam first moment decay (default: 0.9)
        beta2: Adam second moment decay (default: 0.999)
        eps: Numerical stability constant for Adam (default: 1e-8)

    Example:
        >>> optimizer = AdaptiveUPGD(model.parameters(), lr=3e-4)
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
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta_utility < 1.0:
            raise ValueError(f"Invalid beta_utility parameter: {beta_utility}")
        if not 0.0 <= sigma:
            raise ValueError(f"Invalid sigma value: {sigma}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps value: {eps}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        super(AdaptiveUPGD, self).__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(AdaptiveUPGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Perform a single optimization step with adaptive learning rates.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # First pass: compute utilities, moments, and find global maximum
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
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["sec_moment"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Update utility EMA: u = -grad * param
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )

                # Update Adam moments
                first_moment = state["first_moment"]
                sec_moment = state["sec_moment"]
                first_moment.mul_(group["beta1"]).add_(p.grad.data, alpha=1 - group["beta1"])
                sec_moment.mul_(group["beta2"]).add_(p.grad.data ** 2, alpha=1 - group["beta2"])

                # Track global maximum utility
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max.cpu()

        # Second pass: apply updates with adaptive learning rates and scaled utility
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                device = p.device

                # Bias corrections
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]
                bias_correction_beta1 = 1 - group["beta1"] ** state["step"]
                bias_correction_beta2 = 1 - group["beta2"] ** state["step"]

                # Bias-corrected moments (Adam-style)
                exp_avg = state["first_moment"] / bias_correction_beta1
                exp_avg_sq = state["sec_moment"] / bias_correction_beta2

                # Generate perturbation noise
                noise = torch.randn_like(p.grad) * group["sigma"]

                # Scale utility with global normalization
                global_max_on_device = global_max_util.to(device)
                scaled_utility = torch.sigmoid(
                    (state["avg_utility"] / bias_correction_utility) / global_max_on_device
                )

                # Adaptive update with utility-based protection:
                # update = (m / (sqrt(v) + eps) + noise) * (1 - scaled_utility)
                # High utility (important weights) → scaled_utility ≈ 1 → small update
                # Low utility (less important) → scaled_utility ≈ 0 → full update + noise
                adaptive_grad = exp_avg / (exp_avg_sq.sqrt() + group["eps"])
                perturbed_update = (adaptive_grad + noise) * (1 - scaled_utility)

                # Apply weight decay and update
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    perturbed_update,
                    alpha=-2.0 * group["lr"],
                )

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Resets the gradients of all optimized parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zero.
        """
        super().zero_grad(set_to_none=set_to_none)
