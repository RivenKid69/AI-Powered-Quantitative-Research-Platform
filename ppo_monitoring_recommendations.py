"""
PPO Monitoring Recommendations - Add these metrics to detect potential issues

Based on deep audit of PPO implementation, these metrics help detect:
1. Log ratio clamping (potential gradient blocking)
2. Advantage distribution issues
3. Value function clipping behavior
4. Policy entropy collapse
5. BC loss contribution

Add these to distributional_ppo.py in the training loop.
"""

# ============================================================================
# 1. LOG RATIO CLAMPING MONITORING
# ============================================================================
# Location: After line 7871 in distributional_ppo.py
# (after: ratio = torch.exp(log_ratio))

"""
# MONITORING: Track log_ratio clamping to detect gradient blocking
with torch.no_grad():
    # Check how often log_ratio exceeds boundaries
    log_ratio_abs = log_ratio.abs()
    clamp_mask = log_ratio_abs > 20.0

    if clamp_mask.numel() > 0:
        clamp_fraction = clamp_mask.float().mean()
        self.logger.record("train/log_ratio_clamp_frac", float(clamp_fraction))

        # Log statistics of log_ratio
        self.logger.record("train/log_ratio_mean", float(log_ratio.mean()))
        self.logger.record("train/log_ratio_std", float(log_ratio.std()))
        self.logger.record("train/log_ratio_min", float(log_ratio.min()))
        self.logger.record("train/log_ratio_max", float(log_ratio.max()))

        # Log percentage of samples at each boundary
        at_lower_bound = (log_ratio < -19.9).float().mean()
        at_upper_bound = (log_ratio > 19.9).float().mean()
        self.logger.record("train/log_ratio_at_lower_bound", float(at_lower_bound))
        self.logger.record("train/log_ratio_at_upper_bound", float(at_upper_bound))

    # ALERT: If clamp_fraction > 0.01 (1%), investigate!
    # This indicates policy is diverging or learning rate is too high
"""

# ============================================================================
# 2. ADVANTAGE DISTRIBUTION MONITORING
# ============================================================================
# Location: After line 7824 in distributional_ppo.py
# (after: advantages_selected = advantages_normalized_flat[valid_indices])

"""
# MONITORING: Track advantage distribution for normalization issues
with torch.no_grad():
    if advantages_selected.numel() > 0:
        # Raw (pre-normalization) advantage statistics
        adv_raw = advantages_selected_raw  # This is the raw advantage before normalization

        self.logger.record("train/advantage_raw_mean", float(adv_raw.mean()))
        self.logger.record("train/advantage_raw_std", float(adv_raw.std()))
        self.logger.record("train/advantage_raw_min", float(adv_raw.min()))
        self.logger.record("train/advantage_raw_max", float(adv_raw.max()))

        # Normalized advantage statistics (should be ~N(0,1))
        self.logger.record("train/advantage_norm_mean", float(advantages_selected.mean()))
        self.logger.record("train/advantage_norm_std", float(advantages_selected.std()))
        self.logger.record("train/advantage_norm_min", float(advantages_selected.min()))
        self.logger.record("train/advantage_norm_max", float(advantages_selected.max()))

        # Check for anomalies
        # Normalized mean should be close to 0, std close to 1
        if abs(advantages_selected.mean()) > 0.1:
            self.logger.record("warn/advantage_norm_mean_drift", 1.0)
        if abs(advantages_selected.std() - 1.0) > 0.2:
            self.logger.record("warn/advantage_norm_std_drift", 1.0)
"""

# ============================================================================
# 3. VALUE FUNCTION CLIPPING MONITORING (Quantile case)
# ============================================================================
# Location: After line 8446 in distributional_ppo.py
# (after: critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped))

"""
# MONITORING: Track VF clipping behavior
if clip_range_vf_value is not None:
    with torch.no_grad():
        # Log both loss components
        self.logger.record("train/critic_loss_unclipped_quantile",
                          float(critic_loss_unclipped.item()))
        self.logger.record("train/critic_loss_clipped_quantile",
                          float(critic_loss_clipped.item()))

        # How often does clipping matter?
        clipped_active_mask = (critic_loss_clipped > critic_loss_unclipped)
        clipped_active_frac = clipped_active_mask.float().mean()
        self.logger.record("train/vf_clip_active_frac_quantile",
                          float(clipped_active_frac))

        # Log the difference
        loss_diff = (critic_loss_clipped - critic_loss_unclipped).abs()
        self.logger.record("train/vf_clip_loss_diff_quantile", float(loss_diff.item()))

        # Track quantile clipping statistics
        quantiles_clipped_delta = (quantiles_norm_clipped - quantiles_fp32).abs()
        self.logger.record("train/quantiles_clip_delta_mean",
                          float(quantiles_clipped_delta.mean()))
        self.logger.record("train/quantiles_clip_delta_max",
                          float(quantiles_clipped_delta.max()))
"""

# ============================================================================
# 4. VALUE FUNCTION CLIPPING MONITORING (Distributional case)
# ============================================================================
# Location: After line 8730 in distributional_ppo.py
# (after: critic_loss = torch.max(critic_loss, critic_loss_clipped))

"""
# MONITORING: Track VF clipping behavior for distributional case
if clip_range_vf_value is not None:
    with torch.no_grad():
        # Log both loss components
        self.logger.record("train/critic_loss_unclipped_dist",
                          float(critic_loss_unclipped.item()))
        self.logger.record("train/critic_loss_clipped_dist",
                          float(critic_loss_clipped.item()))

        # How often does clipping matter?
        clipped_active = (critic_loss_clipped > critic_loss_unclipped)
        self.logger.record("train/vf_clip_active_dist", 1.0 if clipped_active else 0.0)

        # Log the difference
        loss_diff = (critic_loss_clipped - critic_loss_unclipped).abs()
        self.logger.record("train/vf_clip_loss_diff_dist", float(loss_diff.item()))
"""

# ============================================================================
# 5. POLICY ENTROPY MONITORING
# ============================================================================
# Location: After line 8023 in distributional_ppo.py
# (after: policy_entropy_count += int(entropy_detached.numel()))

"""
# MONITORING: Track entropy distribution for exploration issues
with torch.no_grad():
    if entropy_selected.numel() > 0:
        self.logger.record("train/entropy_min", float(entropy_selected.min()))
        self.logger.record("train/entropy_max", float(entropy_selected.max()))
        self.logger.record("train/entropy_mean", float(entropy_selected.mean()))
        self.logger.record("train/entropy_std", float(entropy_selected.std()))

        # ALERT: If entropy drops below threshold, policy might collapse
        entropy_threshold = 0.01  # Adjust based on action space
        low_entropy_frac = (entropy_selected < entropy_threshold).float().mean()
        self.logger.record("train/low_entropy_frac", float(low_entropy_frac))

        if low_entropy_frac > 0.5:
            self.logger.record("warn/entropy_collapse", 1.0)
"""

# ============================================================================
# 6. BC LOSS CONTRIBUTION MONITORING
# ============================================================================
# Location: After line 7935 in distributional_ppo.py
# (after: policy_loss_bc_weighted = policy_loss_bc * bc_coef)

"""
# MONITORING: Track BC loss contribution relative to PPO loss
if bc_coef > 0.0:
    with torch.no_grad():
        # Ratio of BC loss to PPO loss
        ppo_magnitude = abs(policy_loss_ppo.item())
        bc_magnitude = abs(policy_loss_bc_weighted.item())
        total_magnitude = ppo_magnitude + bc_magnitude

        if total_magnitude > 1e-8:
            bc_ratio = bc_magnitude / total_magnitude
            self.logger.record("train/bc_loss_ratio", float(bc_ratio))

            # ALERT: If BC loss dominates (>0.8), might overwhelm PPO learning
            if bc_ratio > 0.8:
                self.logger.record("warn/bc_loss_dominates", 1.0)

        # Log AWR weight statistics
        self.logger.record("train/awr_weight_mean", float(weights.mean()))
        self.logger.record("train/awr_weight_std", float(weights.std()))
        self.logger.record("train/awr_weight_min", float(weights.min()))
        self.logger.record("train/awr_weight_max", float(weights.max()))

        # Check if max weight is being hit
        max_weight = 100.0
        at_max_weight = (weights > max_weight * 0.99).float().mean()
        self.logger.record("train/awr_at_max_weight_frac", float(at_max_weight))
"""

# ============================================================================
# 7. GRADIENT NORM MONITORING ENHANCEMENT
# ============================================================================
# Location: After line 8827 in distributional_ppo.py
# (after: self.logger.record("train/grad_norm_post_clip", float(post_clip_norm)))

"""
# MONITORING: Enhanced gradient statistics
with torch.no_grad():
    # Compute per-layer gradient norms (useful for debugging vanishing/exploding gradients)
    layer_grad_norms = {}
    for name, param in self.policy.named_parameters():
        if param.grad is not None:
            grad_norm = float(param.grad.norm().item())
            layer_grad_norms[f"grad_norm/{name}"] = grad_norm

    # Log top-5 layers with highest gradient norms
    sorted_layers = sorted(layer_grad_norms.items(), key=lambda x: x[1], reverse=True)
    for i, (name, norm) in enumerate(sorted_layers[:5]):
        self.logger.record(f"train/top_grad_norm_{i}", norm)

    # Compute gradient clipping ratio
    if post_clip_norm > 0:
        clip_ratio = post_clip_norm / float(grad_norm_value)
        self.logger.record("train/grad_clip_ratio", float(clip_ratio))

        # ALERT: If clip_ratio << 1 frequently, gradients are being heavily clipped
        if clip_ratio < 0.5:
            self.logger.record("warn/heavy_grad_clipping", 1.0)
"""

# ============================================================================
# 8. RATIO STATISTICS FOR PPO CLIPPING
# ============================================================================
# Location: After line 7889 in distributional_ppo.py
# (after: self.logger.record("train/clip_fraction_batch", float(clipped.item())))

"""
# MONITORING: Enhanced ratio statistics
with torch.no_grad():
    if torch.any(finite_mask):
        ratio_finite_vals = ratio_detached[finite_mask]

        # Detailed ratio distribution
        self.logger.record("train/ratio_mean", float(ratio_finite_vals.mean()))
        self.logger.record("train/ratio_std", float(ratio_finite_vals.std()))
        self.logger.record("train/ratio_min", float(ratio_finite_vals.min()))
        self.logger.record("train/ratio_max", float(ratio_finite_vals.max()))

        # Quantiles
        if ratio_finite_vals.numel() > 10:
            ratio_sorted = torch.sort(ratio_finite_vals).values
            p05 = ratio_sorted[int(0.05 * len(ratio_sorted))]
            p50 = ratio_sorted[int(0.50 * len(ratio_sorted))]
            p95 = ratio_sorted[int(0.95 * len(ratio_sorted))]

            self.logger.record("train/ratio_p05", float(p05))
            self.logger.record("train/ratio_p50", float(p50))
            self.logger.record("train/ratio_p95", float(p95))

        # Count samples in each clipping region
        below_lower = (ratio_finite_vals < 1 - clip_range).float().mean()
        within_range = ((ratio_finite_vals >= 1 - clip_range) &
                       (ratio_finite_vals <= 1 + clip_range)).float().mean()
        above_upper = (ratio_finite_vals > 1 + clip_range).float().mean()

        self.logger.record("train/ratio_below_lower_bound", float(below_lower))
        self.logger.record("train/ratio_within_bounds", float(within_range))
        self.logger.record("train/ratio_above_upper_bound", float(above_upper))
"""

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
HOW TO ADD THESE METRICS:

1. Open distributional_ppo.py
2. Find the locations indicated in each section above
3. Copy the monitoring code into those locations
4. Run training and monitor logs for:

   a) HIGH PRIORITY ALERTS (investigate immediately):
      - train/log_ratio_clamp_frac > 0.01
      - warn/advantage_norm_mean_drift or warn/advantage_norm_std_drift
      - warn/entropy_collapse
      - warn/bc_loss_dominates
      - warn/heavy_grad_clipping

   b) MEDIUM PRIORITY (monitor trends):
      - train/vf_clip_active_frac (should be 0.1-0.5 in early training)
      - train/bc_loss_ratio (should decrease over time if BC is auxiliary)
      - train/ratio_p05, train/ratio_p95 (should stay near 1.0)

   c) LOW PRIORITY (for debugging):
      - All other metrics (useful when diagnosing specific issues)

5. Create a dashboard (TensorBoard/Weights&Biases) to visualize these metrics
"""

# ============================================================================
# EXPECTED VALUES FOR HEALTHY TRAINING
# ============================================================================

HEALTHY_RANGES = {
    "train/log_ratio_clamp_frac": (0.0, 0.001),  # <0.1% is good
    "train/log_ratio_mean": (-1.0, 1.0),  # Should be near 0
    "train/log_ratio_std": (0.0, 2.0),  # Moderate variance

    "train/advantage_norm_mean": (-0.1, 0.1),  # Should be ~0 after normalization
    "train/advantage_norm_std": (0.8, 1.2),  # Should be ~1 after normalization

    "train/vf_clip_active_frac_quantile": (0.0, 0.5),  # Some clipping is OK
    "train/entropy_mean": (0.1, 5.0),  # Depends on action space

    "train/bc_loss_ratio": (0.0, 0.3),  # BC should be auxiliary, not dominant

    "train/grad_clip_ratio": (0.7, 1.0),  # >0.7 means mild clipping

    "train/ratio_mean": (0.9, 1.1),  # Should stay close to 1
    "train/ratio_p05": (0.7, 0.95),  # Lower bound
    "train/ratio_p95": (1.05, 1.3),  # Upper bound
}

"""
Use these ranges to create alerts in your monitoring system.
Example with Weights & Biases:

import wandb

for metric, (min_val, max_val) in HEALTHY_RANGES.items():
    wandb.alert(
        title=f"{metric} out of range",
        text=f"{metric} outside healthy range [{min_val}, {max_val}]",
        level=wandb.AlertLevel.WARN,
        wait_duration=timedelta(minutes=5)
    )
"""
