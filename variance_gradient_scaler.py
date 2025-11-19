"""
Variance Gradient Scaler

Implements adaptive gradient scaling based on gradient variance statistics.

This module provides a mechanism to monitor and adaptively scale gradients
during training based on their variance, which can improve training stability
and convergence. Based on research from "A Study of Gradient Variance in
Deep Learning" (Faghri et al., 2020).

Key features:
- Tracks gradient variance across training steps using exponential moving average
- Computes normalized gradient variance metric
- Adaptively scales gradients to reduce variance and improve stability
- Provides comprehensive logging for monitoring

Algorithm:
    1. Compute gradient statistics (mean, variance, norm) per parameter
    2. Track exponential moving average of statistics with bias correction
    3. Compute normalized gradient variance: Var[g] / (E[g]^2 + eps)
    4. Apply adaptive scaling: g_scaled = g / (1 + alpha * normalized_var)
    5. Log metrics for analysis

Hyperparameters:
    - enabled: Enable/disable variance scaling (default: True)
    - beta: EMA decay rate for gradient statistics (default: 0.99)
    - alpha: Scaling strength coefficient (default: 0.1)
    - eps: Numerical stability epsilon (default: 1e-8)
    - warmup_steps: Number of steps before applying scaling (default: 100)
"""

import torch
from typing import Optional, Dict, Any, Iterable, List


class VarianceGradientScaler:
    """
    Adaptive gradient scaler based on variance statistics.

    Monitors gradient variance during training and applies adaptive scaling
    to reduce variance and improve training stability.

    Args:
        parameters: Iterable of parameters to track gradients for
        enabled: Whether to apply gradient scaling (default: True)
        beta: EMA decay rate for gradient statistics (default: 0.99)
        alpha: Scaling strength coefficient (default: 0.1)
        eps: Numerical stability epsilon (default: 1e-8)
        warmup_steps: Number of steps before applying scaling (default: 100)
        track_per_param: Track per-parameter statistics (default: False)
        logger: Optional logger for metrics

    Example:
        >>> scaler = VarianceGradientScaler(model.parameters(), enabled=True)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> scaler.scale_gradients()  # Apply variance-based scaling
        >>> torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        >>> optimizer.step()
        >>> scaler.step()  # Update statistics
    """

    def __init__(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        enabled: bool = True,
        beta: float = 0.99,
        alpha: float = 0.1,
        eps: float = 1e-8,
        warmup_steps: int = 100,
        track_per_param: bool = False,
        logger: Optional[Any] = None,
    ) -> None:
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

        self.enabled = bool(enabled)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.warmup_steps = int(warmup_steps)
        self.track_per_param = bool(track_per_param)
        self._logger = logger

        # Global statistics (aggregated across all parameters)
        self._step_count: int = 0
        self._grad_mean_ema: Optional[float] = None
        self._grad_var_ema: Optional[float] = None
        self._grad_norm_ema: Optional[float] = None
        self._grad_max_ema: Optional[float] = None

        # Per-parameter statistics (optional)
        self._param_stats: Dict[int, Dict[str, torch.Tensor]] = {}

        # Cached parameters list
        self._parameters: Optional[List[torch.nn.Parameter]] = None
        if parameters is not None:
            self._parameters = list(parameters)

    def update_parameters(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Update the list of parameters to track."""
        self._parameters = list(parameters)

    def _log(self, key: str, value: float) -> None:
        """Log a metric value if logger is available."""
        if self._logger is None:
            return
        record = getattr(self._logger, "record", None)
        if callable(record):
            try:
                record(key, float(value))
            except Exception:
                pass

    @torch.no_grad()
    def compute_gradient_statistics(self) -> Dict[str, float]:
        """
        Compute current gradient statistics across all parameters.

        Returns:
            Dictionary containing:
                - grad_norm: Global gradient norm (L2)
                - grad_mean: Mean of absolute gradient values
                - grad_var: Variance of gradient values
                - grad_max: Maximum absolute gradient value
                - num_params: Number of parameters with gradients
        """
        if self._parameters is None:
            return {
                "grad_norm": 0.0,
                "grad_mean": 0.0,
                "grad_var": 0.0,
                "grad_max": 0.0,
                "num_params": 0,
            }

        grad_norms_sq = []
        grad_values = []
        grad_max = 0.0
        num_params = 0

        for param in self._parameters:
            if param.grad is None:
                continue

            grad = param.grad.data
            grad_norms_sq.append(grad.pow(2).sum().item())
            grad_values.append(grad.abs().flatten())
            grad_max = max(grad_max, grad.abs().max().item())
            num_params += 1

        if num_params == 0:
            return {
                "grad_norm": 0.0,
                "grad_mean": 0.0,
                "grad_var": 0.0,
                "grad_max": 0.0,
                "num_params": 0,
            }

        # Compute global gradient norm
        grad_norm = (sum(grad_norms_sq) ** 0.5)

        # Compute mean and variance of gradient values
        # FIXED: Use abs() for both mean and variance for mathematical consistency
        all_grads = torch.cat(grad_values)  # grad_values already contains abs values
        grad_mean = all_grads.mean().item()
        grad_var = all_grads.var().item()

        return {
            "grad_norm": grad_norm,
            "grad_mean": grad_mean,
            "grad_var": grad_var,
            "grad_max": grad_max,
            "num_params": num_params,
        }

    @torch.no_grad()
    def update_statistics(self) -> None:
        """Update exponential moving averages of gradient statistics."""
        stats = self.compute_gradient_statistics()

        if stats["num_params"] == 0:
            return

        # Initialize on first step
        if self._grad_mean_ema is None:
            self._grad_mean_ema = stats["grad_mean"]
            self._grad_var_ema = stats["grad_var"]
            self._grad_norm_ema = stats["grad_norm"]
            self._grad_max_ema = stats["grad_max"]
        else:
            # Update EMA
            self._grad_mean_ema = self.beta * self._grad_mean_ema + (1 - self.beta) * stats["grad_mean"]
            self._grad_var_ema = self.beta * self._grad_var_ema + (1 - self.beta) * stats["grad_var"]
            self._grad_norm_ema = self.beta * self._grad_norm_ema + (1 - self.beta) * stats["grad_norm"]
            self._grad_max_ema = self.beta * self._grad_max_ema + (1 - self.beta) * stats["grad_max"]

    @torch.no_grad()
    def get_normalized_variance(self) -> float:
        """
        Compute normalized gradient variance: Var[|g|] / (E[|g|]^2 + eps)

        This metric is scale-invariant and indicates the relative variability
        of gradients. Higher values suggest more unstable gradients.

        Returns:
            Normalized gradient variance (0.0 if statistics not available)
        """
        if self._grad_var_ema is None or self._grad_mean_ema is None:
            return 0.0

        # FIXED: Bias correction using correct step count (without +1)
        # since step_count is incremented AFTER update_statistics in step()
        bias_correction = 1.0 - self.beta ** self._step_count if self._step_count > 0 else 1.0
        var_corrected = self._grad_var_ema / bias_correction
        mean_corrected = self._grad_mean_ema / bias_correction

        # Normalized variance with numerical stability
        # Ensure denominator is not too small
        denominator = max(mean_corrected ** 2, 1e-12) + self.eps
        normalized_var = var_corrected / denominator

        # FIXED: Protection against inf/nan
        if not (normalized_var >= 0.0 and normalized_var < float('inf')):
            return 0.0

        # FIXED: Clip extreme values to prevent numerical issues
        normalized_var = min(normalized_var, 1e6)

        return float(normalized_var)

    @torch.no_grad()
    def get_scaling_factor(self) -> float:
        """
        Compute gradient scaling factor based on normalized variance.

        Scaling factor: 1 / (1 + alpha * normalized_var)

        This reduces gradients when variance is high, improving stability.
        During warmup, returns 1.0 (no scaling).

        Returns:
            Scaling factor in range (0, 1]
        """
        if not self.enabled or self._step_count < self.warmup_steps:
            return 1.0

        normalized_var = self.get_normalized_variance()
        scaling_factor = 1.0 / (1.0 + self.alpha * normalized_var)

        # FIXED: Ensure scaling factor is in valid range and not too small
        # Prevent gradients from becoming zero
        scaling_factor = max(scaling_factor, 1e-4)
        scaling_factor = min(scaling_factor, 1.0)

        return float(scaling_factor)

    @torch.no_grad()
    def scale_gradients(self) -> float:
        """
        Apply variance-based scaling to gradients.

        Scales all parameter gradients by the computed scaling factor.
        Should be called after loss.backward() but before optimizer.step().

        Returns:
            The scaling factor applied (1.0 if not enabled or during warmup)
        """
        if self._parameters is None:
            return 1.0

        scaling_factor = self.get_scaling_factor()

        if scaling_factor < 1.0:
            for param in self._parameters:
                if param.grad is not None:
                    param.grad.data.mul_(scaling_factor)

        return scaling_factor

    @torch.no_grad()
    def step(self) -> None:
        """
        Update scaler state after optimizer step.

        Should be called after optimizer.step() to update statistics
        and log metrics.
        """
        # FIXED: Increment step count BEFORE update for correct bias correction
        self._step_count += 1
        self.update_statistics()

        # Log metrics
        if self._grad_norm_ema is not None:
            bias_correction = 1.0 - self.beta ** self._step_count

            self._log("vgs/grad_norm_ema", self._grad_norm_ema / bias_correction)
            self._log("vgs/grad_mean_ema", self._grad_mean_ema / bias_correction)
            self._log("vgs/grad_var_ema", self._grad_var_ema / bias_correction)
            self._log("vgs/grad_max_ema", self._grad_max_ema / bias_correction)
            self._log("vgs/normalized_variance", self.get_normalized_variance())
            self._log("vgs/scaling_factor", self.get_scaling_factor())
            self._log("vgs/step_count", float(self._step_count))
            self._log("vgs/warmup_active", float(self._step_count < self.warmup_steps))

    @torch.no_grad()
    def reset_statistics(self) -> None:
        """Reset all accumulated statistics."""
        self._step_count = 0
        self._grad_mean_ema = None
        self._grad_var_ema = None
        self._grad_norm_ema = None
        self._grad_max_ema = None
        self._param_stats.clear()

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "beta": self.beta,
            "alpha": self.alpha,
            "eps": self.eps,
            "warmup_steps": self.warmup_steps,
            "step_count": self._step_count,
            "grad_mean_ema": self._grad_mean_ema,
            "grad_var_ema": self._grad_var_ema,
            "grad_norm_ema": self._grad_norm_ema,
            "grad_max_ema": self._grad_max_ema,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        self.enabled = state_dict.get("enabled", self.enabled)
        self.beta = state_dict.get("beta", self.beta)
        self.alpha = state_dict.get("alpha", self.alpha)
        self.eps = state_dict.get("eps", self.eps)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self._step_count = state_dict.get("step_count", 0)
        self._grad_mean_ema = state_dict.get("grad_mean_ema", None)
        self._grad_var_ema = state_dict.get("grad_var_ema", None)
        self._grad_norm_ema = state_dict.get("grad_norm_ema", None)
        self._grad_max_ema = state_dict.get("grad_max_ema", None)

    def __repr__(self) -> str:
        return (
            f"VarianceGradientScaler("
            f"enabled={self.enabled}, "
            f"beta={self.beta}, "
            f"alpha={self.alpha}, "
            f"eps={self.eps}, "
            f"warmup_steps={self.warmup_steps}, "
            f"step_count={self._step_count})"
        )
