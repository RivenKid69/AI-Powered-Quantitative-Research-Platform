"""
UPGDW Optimizer (UPGD with Decoupled Weight Decay)

Implements UPGD with AdamW-style decoupled weight decay for better regularization.

This is the recommended optimizer for the TradingBot2 project as it:
1. Matches the existing AdamW optimizer interface
2. Provides utility-based weight protection
3. Uses decoupled weight decay (applied directly to params, not gradients)
4. Includes adaptive learning rates from Adam

The key difference from AdaptiveUPGD is that weight decay is applied directly
to parameters before the gradient update, following the AdamW approach which
has been shown to improve generalization.

Reference:
    Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization.
    ICLR 2019. https://arxiv.org/abs/1711.05101
"""

import torch
from typing import Any, Dict, Iterable, Optional, Union


class UPGDW(torch.optim.Optimizer):
    """
    UPGD with decoupled weight decay (AdamW-style).

    This optimizer combines utility-based weight protection with Adam adaptive
    learning rates and decoupled weight decay for improved regularization.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-4)
        betas: Coefficients for computing running averages of gradient and its square
               (beta1, beta2) where beta2 also serves as beta_utility (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Decoupled weight decay coefficient (default: 0.01)
        sigma: Standard deviation of perturbation noise (default: 0.001)
        amsgrad: Whether to use AMSGrad variant (not implemented, for compatibility)

    Example:
        >>> # Drop-in replacement for AdamW
        >>> optimizer = UPGDW(model.parameters(), lr=1e-4, weight_decay=0.01)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        sigma: float = 0.001,
        amsgrad: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= sigma:
            raise ValueError(f"Invalid sigma value: {sigma}")

        if amsgrad:
            raise NotImplementedError("AMSGrad variant is not implemented for UPGDW")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sigma=sigma,
            amsgrad=amsgrad,
        )
        super(UPGDW, self).__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(UPGDW, self).__setstate__(state)

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

        # First pass: compute utilities, moments, and find global maximum
        # ═══════════════════════════════════════════════════════════════════════════
        # НЕ БАГ: ИНИЦИАЛИЗАЦИЯ -inf ДЛЯ global_max_util
        # ═══════════════════════════════════════════════════════════════════════════
        # Если global_max_util остаётся -inf, это означает что ВСЕ параметры имели
        # grad=None в первом проходе. Но тогда они ТАКЖЕ будут пропущены во втором
        # проходе (if p.grad is None: continue), поэтому деление на -inf не произойдёт.
        #
        # Сценарий "gradients появляются между проходами" невозможен — оба прохода
        # итерируют по одним и тем же параметрам синхронно в рамках одного step().
        # Reference: CLAUDE.md → "НЕ БАГИ" → #19
        # ═══════════════════════════════════════════════════════════════════════════
        global_max_util = torch.tensor(-torch.inf, device="cpu")

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Update utility EMA using beta2 (same as second moment decay)
                avg_utility = state["avg_utility"]
                avg_utility.mul_(beta2).add_(
                    -p.grad.data * p.data, alpha=1 - beta2
                )

                # Update Adam moments
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(p.grad.data ** 2, alpha=1 - beta2)

                # Track global maximum utility
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max.cpu()

        # Second pass: apply decoupled weight decay and adaptive updates
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                device = p.device

                # Decoupled weight decay (AdamW-style)
                # Applied before gradient update, not added to gradients
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Bias corrections
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Bias-corrected moments
                exp_avg_corrected = state["exp_avg"] / bias_correction1
                exp_avg_sq_corrected = state["exp_avg_sq"] / bias_correction2

                # Generate perturbation noise
                noise = torch.randn_like(p.grad) * group["sigma"]

                # Scale utility with global normalization
                global_max_on_device = global_max_util.to(device)
                scaled_utility = torch.sigmoid(
                    (state["avg_utility"] / bias_correction2) / global_max_on_device
                )

                # Adaptive gradient with utility-based protection
                # High utility → small update (protect important weights)
                # Low utility → full update with noise (maintain plasticity)
                denom = exp_avg_sq_corrected.sqrt().add_(group["eps"])
                step_size = group["lr"]

                # Update: param -= lr * (m / sqrt(v) + noise) * (1 - utility)
                update = (exp_avg_corrected / denom + noise) * (1 - scaled_utility)
                p.data.add_(update, alpha=-step_size)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Resets the gradients of all optimized parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zero.
        """
        super().zero_grad(set_to_none=set_to_none)
