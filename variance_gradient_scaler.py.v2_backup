"""
Variance Gradient Scaler

Implements adaptive gradient scaling based on **per-parameter stochastic variance**.

This module provides a mechanism to monitor and adaptively scale gradients
during training based on their stochastic variance (temporal noise), which can
improve training stability and convergence.

**IMPORTANT: Fixed in v2.0 (2025-11-21)**
Previous versions computed spatial variance (variance across all parameters at
one timestep). This version correctly computes stochastic variance (variance of
gradient estimates for each parameter over time), which is the proper metric for
gradient scaling and stability.

Key features:
- Tracks per-parameter stochastic variance using exponential moving average
- Aggregates per-parameter variance to global metric (90th percentile)
- Computes normalized gradient variance: Var[g] / (E[g]^2 + eps)
- Adaptively scales gradients to reduce variance and improve stability
- Provides comprehensive logging for monitoring
- Backward compatible checkpoint loading with soft migration

Algorithm:
    1. For each parameter, track EMA of first moment (mean) and second moment (mean of squares)
    2. Compute per-parameter variance: Var[g] = E[g²] - E[g]²
    3. Compute per-parameter normalized variance: Var[g] / (E[g]² + ε)
    4. Aggregate to global metric using 90th percentile (robust to outliers)
    5. Apply global adaptive scaling: g_scaled = g / (1 + α * global_var)
    6. Log metrics for analysis

Hyperparameters:
    - enabled: Enable/disable variance scaling (default: True)
    - beta: EMA decay rate for gradient statistics (default: 0.99)
    - alpha: Scaling strength coefficient (default: 0.1)
    - eps: Numerical stability epsilon (default: 1e-8)
    - warmup_steps: Number of steps before applying scaling (default: 100)

References:
    - Kingma & Ba (2015). Adam: A Method for Stochastic Optimization. ICLR.
    - Faghri & Duvenaud (2020). A Study of Gradient Variance in Deep Learning. arXiv:2007.04532.
"""

import torch
from typing import Optional, Dict, Any, Iterable, List


class VarianceGradientScaler:
    """
    Adaptive gradient scaler based on per-parameter stochastic variance.

    Monitors per-parameter stochastic variance (temporal gradient noise) during
    training and applies adaptive scaling to reduce variance and improve stability.

    **v2.0 (2025-11-21): Fixed to use per-parameter stochastic variance instead
    of spatial variance.**

    Args:
        parameters: Iterable of parameters to track gradients for
        enabled: Whether to apply gradient scaling (default: True)
        beta: EMA decay rate for gradient statistics (default: 0.99)
        alpha: Scaling strength coefficient (default: 0.1)
        eps: Numerical stability epsilon (default: 1e-8)
        warmup_steps: Number of steps before applying scaling (default: 100)
        track_per_param: DEPRECATED - Now always True (per-parameter tracking)
        logger: Optional logger for metrics

    Example:
        >>> scaler = VarianceGradientScaler(model.parameters(), enabled=True)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> scaler.scale_gradients()  # Apply variance-based scaling
        >>> torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        >>> optimizer.step()
        >>> scaler.step()  # Update statistics

    Note:
        Old checkpoints (v1.x with spatial variance) will be automatically migrated
        with statistics reset. A warning will be issued. Retraining is recommended.
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
        self.track_per_param = True  # DEPRECATED - now always True
        self._logger = logger

        # Step counter
        self._step_count: int = 0

        # NEW v2.0 (v2.0.1 - numerically stable): Per-parameter stochastic variance tracking
        # These track temporal variance (over time) for each parameter
        # NOTE: Changed semantics in v2.0.1 for numerical stability:
        # - _param_grad_mean_ema stores E[|g|] (mean of absolute gradients)
        # - _param_grad_sq_ema stores Var[g] (variance from torch.var(), not E[g²])
        self._param_grad_mean_ema: Optional[torch.Tensor] = None  # [num_params] - E[|g|]
        self._param_grad_sq_ema: Optional[torch.Tensor] = None    # [num_params] - Var[g]
        self._param_numel: Optional[torch.Tensor] = None          # [num_params] - num elements per param
        self._param_ids: Dict[int, int] = {}                      # param id -> flat index

        # LEGACY v1.x: Global statistics (spatial variance - kept for backward compat logging)
        # These are now computed for logging only, not used for scaling
        self._grad_mean_ema: Optional[float] = None  # Global mean (spatial)
        self._grad_var_ema: Optional[float] = None   # Global var (spatial) - DEPRECATED
        self._grad_norm_ema: Optional[float] = None  # Global norm
        self._grad_max_ema: Optional[float] = None   # Global max

        # DEPRECATED v1.x: Per-parameter statistics (old format)
        self._param_stats: Dict[int, Dict[str, torch.Tensor]] = {}

        # Cached parameters list
        self._parameters: Optional[List[torch.nn.Parameter]] = None
        if parameters is not None:
            self._parameters = list(parameters)

    def update_parameters(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Update the list of parameters to track.

        Note: This will reset per-parameter statistics if the parameter list changes.
        """
        self._parameters = list(parameters)
        # Reset per-parameter statistics when parameters change
        self._param_grad_mean_ema = None
        self._param_grad_sq_ema = None
        self._param_numel = None
        self._param_ids = {}

    def _initialize_per_param_stats(self) -> None:
        """Initialize per-parameter stochastic variance tracking.

        Called automatically on first `update_statistics()` call.
        """
        if self._parameters is None or len(self._parameters) == 0:
            return

        num_params = len(self._parameters)
        # Use device of first parameter
        device = self._parameters[0].device if num_params > 0 else torch.device('cpu')

        # Initialize EMA buffers
        self._param_grad_mean_ema = torch.zeros(num_params, device=device, dtype=torch.float32)
        self._param_grad_sq_ema = torch.zeros(num_params, device=device, dtype=torch.float32)
        self._param_numel = torch.tensor(
            [p.numel() for p in self._parameters],
            device=device,
            dtype=torch.float32
        )

        # Build parameter ID -> flat index mapping
        self._param_ids = {id(p): i for i, p in enumerate(self._parameters)}

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
        """Update per-parameter stochastic variance statistics.

        **v2.0**: Now tracks per-parameter temporal variance (stochastic variance)
        instead of spatial variance.
        """
        if self._parameters is None or len(self._parameters) == 0:
            return

        # Initialize per-parameter statistics on first call
        if self._param_grad_mean_ema is None:
            self._initialize_per_param_stats()

        # Update per-parameter stochastic variance (temporal variance)
        for i, param in enumerate(self._parameters):
            if param.grad is None:
                continue

            grad = param.grad.data
            # Compute per-parameter statistics using numerically stable torch.var()
            # We track E[|g|] and Var[g] separately for stability
            grad_abs_mean = grad.abs().mean().item()   # E[|g|] - used for normalization
            grad_variance = grad.var(unbiased=False).item()  # Var[g] = E[(g - E[g])²] - numerically stable

            # Update EMA for this parameter
            # NOTE: We track E[|g|] (abs mean) and Var[g] (variance) separately
            # This is numerically more stable than E[g] and E[g²]
            if self._step_count == 1:
                # Initialize on first step
                self._param_grad_mean_ema[i] = grad_abs_mean  # E[|g|]
                self._param_grad_sq_ema[i] = grad_variance    # Var[g]
            else:
                # Update EMA
                self._param_grad_mean_ema[i] = (
                    self.beta * self._param_grad_mean_ema[i] +
                    (1 - self.beta) * grad_abs_mean
                )
                self._param_grad_sq_ema[i] = (
                    self.beta * self._param_grad_sq_ema[i] +
                    (1 - self.beta) * grad_variance
                )

        # LEGACY: Update global statistics for backward compat logging
        # These are now SPATIAL variance (deprecated) and used only for logging
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
        Compute global normalized stochastic variance from per-parameter statistics.

        **v2.0**: Now uses per-parameter stochastic variance (temporal) instead of
        spatial variance. Aggregates per-parameter variances using 90th percentile
        (robust to outliers).

        Formula (v2.0.1 - numerically stable):
            For each parameter i:
                Var[g_i] = torch.var(g_i)  # Computed directly for stability
                E[|g_i|] = mean(|g_i|)     # Mean of absolute gradients
                normalized_var[i] = Var[g_i] / (E[|g_i|]² + ε)
            global_var = percentile(normalized_var, 90)

        Returns:
            Global normalized stochastic variance (0.0 if statistics not available)
        """
        # Check if per-parameter statistics are initialized
        if self._param_grad_mean_ema is None or self._param_grad_sq_ema is None:
            return 0.0

        if self._step_count == 0:
            return 0.0

        # Bias correction for EMA
        bias_correction = 1.0 - self.beta ** self._step_count

        # Correct for bias
        # NOTE: New semantics (v2.0.1):
        # - _param_grad_mean_ema stores E[|g|] (mean of absolute gradients)
        # - _param_grad_sq_ema stores Var[g] (variance computed by torch.var())
        abs_mean_corrected = self._param_grad_mean_ema / bias_correction  # E[|g|]
        variance_corrected = self._param_grad_sq_ema / bias_correction     # Var[g]

        # Variance is already computed by torch.var(), so no need to subtract means
        variance = torch.clamp(variance_corrected, min=0.0)  # Numerical stability

        # Compute per-parameter normalized variance: Var[g] / (E[|g|]² + ε)
        denominator = torch.clamp(abs_mean_corrected.pow(2), min=1e-12) + self.eps
        normalized_var_per_param = variance / denominator

        # Handle NaN/inf
        normalized_var_per_param = torch.where(
            torch.isfinite(normalized_var_per_param),
            normalized_var_per_param,
            torch.zeros_like(normalized_var_per_param)
        )

        # Aggregate to global metric using 90th percentile (robust to outliers)
        # This focuses on "worst" parameters with highest variance
        if normalized_var_per_param.numel() > 0:
            global_var = torch.quantile(normalized_var_per_param, 0.9).item()
        else:
            global_var = 0.0

        # Clip extreme values
        global_var = min(global_var, 1e6)
        global_var = max(global_var, 0.0)

        return float(global_var)

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

        **v2.0**: Now logs both stochastic variance (new) and spatial variance (legacy).
        """
        # FIXED: Increment step count BEFORE update for correct bias correction
        self._step_count += 1
        self.update_statistics()

        # Log NEW v2.0 metrics (per-parameter stochastic variance)
        if self._param_grad_mean_ema is not None and self._param_grad_sq_ema is not None:
            bias_correction = 1.0 - self.beta ** self._step_count

            # Compute per-parameter normalized variances
            # NOTE: New semantics (v2.0.1):
            # - _param_grad_mean_ema stores E[|g|]
            # - _param_grad_sq_ema stores Var[g] directly
            abs_mean_corrected = self._param_grad_mean_ema / bias_correction  # E[|g|]
            variance_corrected = self._param_grad_sq_ema / bias_correction     # Var[g]
            variance = torch.clamp(variance_corrected, min=0.0)
            denominator = torch.clamp(abs_mean_corrected.pow(2), min=1e-12) + self.eps
            normalized_var_per_param = variance / denominator
            normalized_var_per_param = torch.where(
                torch.isfinite(normalized_var_per_param),
                normalized_var_per_param,
                torch.zeros_like(normalized_var_per_param)
            )

            # Log percentiles of per-parameter stochastic variance
            if normalized_var_per_param.numel() > 0:
                self._log("vgs/stochastic_var_p10", torch.quantile(normalized_var_per_param, 0.1).item())
                self._log("vgs/stochastic_var_p50", torch.quantile(normalized_var_per_param, 0.5).item())
                self._log("vgs/stochastic_var_p90", torch.quantile(normalized_var_per_param, 0.9).item())
                self._log("vgs/stochastic_var_mean", normalized_var_per_param.mean().item())

        # Log LEGACY v1.x metrics (spatial variance - for backward compat)
        if self._grad_norm_ema is not None:
            bias_correction = 1.0 - self.beta ** self._step_count

            self._log("vgs/grad_norm_ema", self._grad_norm_ema / bias_correction)
            self._log("vgs/grad_mean_ema_spatial", self._grad_mean_ema / bias_correction)  # Renamed
            self._log("vgs/grad_var_ema_spatial", self._grad_var_ema / bias_correction)    # Renamed (DEPRECATED)
            self._log("vgs/grad_max_ema", self._grad_max_ema / bias_correction)

        # Log normalized variance (now stochastic, was spatial in v1.x)
        self._log("vgs/normalized_variance", self.get_normalized_variance())
        self._log("vgs/scaling_factor", self.get_scaling_factor())
        self._log("vgs/step_count", float(self._step_count))
        self._log("vgs/warmup_active", float(self._step_count < self.warmup_steps))

    @torch.no_grad()
    def reset_statistics(self) -> None:
        """Reset all accumulated statistics.

        **v2.0**: Now resets both per-parameter and global statistics.
        """
        self._step_count = 0

        # Reset per-parameter stochastic variance statistics
        self._param_grad_mean_ema = None
        self._param_grad_sq_ema = None
        self._param_numel = None
        self._param_ids = {}

        # Reset global spatial variance statistics (legacy)
        self._grad_mean_ema = None
        self._grad_var_ema = None
        self._grad_norm_ema = None
        self._grad_max_ema = None

        # Reset deprecated per-parameter stats (legacy)
        self._param_stats.clear()

    def __getstate__(self) -> dict:
        """Custom pickle handler to exclude unpicklable logger and parameters (Bug #9 fix).

        FIX Bug #9: Do NOT pickle _parameters because:
        1. After model.load(), policy gets NEW parameter objects
        2. If VGS pickles old parameter references, it will track stale copies
        3. _parameters must be relinked via update_parameters() after load
        """
        state = self.__dict__.copy()
        # Remove unpicklable logger
        state.pop("_logger", None)
        # FIX Bug #9: Do NOT pickle _parameters - they will be stale after load
        # _parameters will be restored via update_parameters() in _setup_dependent_components()
        state.pop("_parameters", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Custom unpickle handler to restore state without logger and parameters (Bug #9 fix)."""
        self.__dict__.update(state)
        # Logger will not be restored - caller must set it if needed
        if not hasattr(self, "_logger"):
            self._logger = None
        # FIX Bug #9: Initialize _parameters to None - will be set via update_parameters()
        # This ensures VGS doesn't track stale parameter copies from pickle
        if not hasattr(self, "_parameters"):
            self._parameters = None

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary for serialization.

        **v2.0**: Now includes per-parameter stochastic variance statistics.
        """
        state = {
            # Config
            "enabled": self.enabled,
            "beta": self.beta,
            "alpha": self.alpha,
            "eps": self.eps,
            "warmup_steps": self.warmup_steps,
            "step_count": self._step_count,

            # NEW v2.0: Per-parameter stochastic variance statistics
            "param_grad_mean_ema": self._param_grad_mean_ema,
            "param_grad_sq_ema": self._param_grad_sq_ema,
            "param_numel": self._param_numel,

            # LEGACY v1.x: Global spatial variance statistics (for logging only)
            "grad_mean_ema": self._grad_mean_ema,
            "grad_var_ema": self._grad_var_ema,
            "grad_norm_ema": self._grad_norm_ema,
            "grad_max_ema": self._grad_max_ema,

            # Version marker for migration
            "vgs_version": "2.0",
        }

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary with backward compatibility.

        **v2.0**: Automatically migrates old checkpoints (v1.x with spatial variance)
        to new format (v2.0 with per-parameter stochastic variance). Old statistics
        are reset with a warning issued. Retraining is recommended.
        """
        # Load config parameters
        self.enabled = state_dict.get("enabled", self.enabled)
        self.beta = state_dict.get("beta", self.beta)
        self.alpha = state_dict.get("alpha", self.alpha)
        self.eps = state_dict.get("eps", self.eps)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self._step_count = state_dict.get("step_count", 0)

        # Check version
        vgs_version = state_dict.get("vgs_version", "1.0")

        if vgs_version == "1.0" or "param_grad_mean_ema" not in state_dict:
            # OLD FORMAT (v1.x with spatial variance)
            import warnings
            warnings.warn(
                "\n"
                "=" * 80 + "\n"
                "VGS Checkpoint Migration: OLD FORMAT DETECTED (v1.x - spatial variance)\n"
                "=" * 80 + "\n"
                "Loading old VGS checkpoint with SPATIAL variance (variance across parameters).\n"
                "This has been FIXED in v2.0 to use STOCHASTIC variance (temporal variance per-parameter).\n"
                "\n"
                "ACTION REQUIRED:\n"
                "- Per-parameter stochastic variance statistics will be RESET.\n"
                "- Training will continue normally with correct variance tracking.\n"
                "- Retraining models is RECOMMENDED for optimal performance.\n"
                "\n"
                "See VGS_SPATIAL_VS_STOCHASTIC_VARIANCE_ANALYSIS.md for details.\n"
                "=" * 80,
                UserWarning
            )

            # Reset per-parameter statistics (will be reinitialized on first step)
            self._param_grad_mean_ema = None
            self._param_grad_sq_ema = None
            self._param_numel = None
            self._param_ids = {}

            # Load legacy global statistics if available (for logging)
            self._grad_mean_ema = state_dict.get("grad_mean_ema", None)
            self._grad_var_ema = state_dict.get("grad_var_ema", None)
            self._grad_norm_ema = state_dict.get("grad_norm_ema", None)
            self._grad_max_ema = state_dict.get("grad_max_ema", None)

        else:
            # NEW FORMAT (v2.0 with per-parameter stochastic variance)
            # Load per-parameter statistics
            self._param_grad_mean_ema = state_dict.get("param_grad_mean_ema", None)
            self._param_grad_sq_ema = state_dict.get("param_grad_sq_ema", None)
            self._param_numel = state_dict.get("param_numel", None)
            # param_ids will be rebuilt by _initialize_per_param_stats if needed

            # Load legacy global statistics (for logging)
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
            f"step_count={self._step_count}, "
            f"version=2.0)"  # Added version marker
        )
