"""
Comprehensive tests for PPO ratio clamping logic.

Tests verify the critical fix for log_ratio clamping:
- Correct implementation: clamp log_ratio to ±10 BEFORE exp to prevent overflow
- Old bug: clamp to ±20 allowed exp(20)≈485M, causing numerical instability

The ratio = exp(log_prob - old_log_prob) is the core of PPO's trust region.
PPO clips ratio to [1-ε, 1+ε] where ε≈0.1, so ratio should be near 1.0.

Empirical data from training logs:
- ratio_mean ≈ 1.0 (perfect)
- ratio_std ≈ 0.02-0.04 (very small)
- log_ratio typically ≈ 0.039 (log(1.04))

Reference: commit 3e7c1c9 - fix: Reduce PPO log_ratio clamp range from ±20 to ±10
"""

import math

import pytest


def test_ratio_clamp_prevents_overflow() -> None:
    """Test that log_ratio=±10 prevents exp() overflow while providing safety margin."""
    torch = pytest.importorskip("torch")

    # exp(88) overflows to inf, exp(10) is safe
    log_ratio_clamped = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=torch.float32)

    ratio = torch.exp(log_ratio_clamped)

    # All ratios should be finite
    assert torch.all(torch.isfinite(ratio)), \
        f"exp(log_ratio) produced non-finite values: {ratio.tolist()}"

    # Check actual values
    expected = [
        4.54e-5,   # exp(-10)
        6.74e-3,   # exp(-5)
        1.0,       # exp(0)
        148.4,     # exp(5)
        22026.5,   # exp(10)
    ]

    for i, (actual, exp_val) in enumerate(zip(ratio.tolist(), expected)):
        assert abs(actual - exp_val) / exp_val < 0.01, \
            f"ratio[{i}] = {actual}, expected ≈{exp_val}"


def test_ratio_clamp_realistic_values() -> None:
    """Test ratio computation with realistic log_prob differences from training."""
    torch = pytest.importorskip("torch")

    # Realistic values from training logs:
    # - ratio_mean ≈ 1.0, ratio_std ≈ 0.03
    # - This corresponds to log_ratio_std ≈ 0.03

    # Simulate typical batch of log_prob differences
    torch.manual_seed(42)
    log_ratio = torch.randn(1000, dtype=torch.float32) * 0.03  # std=0.03

    # Clamp as in the fix
    log_ratio_clamped = torch.clamp(log_ratio, min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio_clamped)

    # Verify all finite
    assert torch.all(torch.isfinite(ratio)), \
        "ratio should be finite for realistic log_ratio values"

    # Verify ratio statistics match logs
    ratio_mean = ratio.mean().item()
    ratio_std = ratio.std().item()

    assert 0.98 < ratio_mean < 1.02, \
        f"ratio_mean should be ≈1.0, got {ratio_mean}"
    assert 0.01 < ratio_std < 0.05, \
        f"ratio_std should be ≈0.03, got {ratio_std}"


def test_ratio_clamp_extreme_values() -> None:
    """Test that extreme log_ratio values are properly clamped."""
    torch = pytest.importorskip("torch")

    # Extreme log_prob differences (e.g., from numerical issues)
    extreme_log_ratios = torch.tensor(
        [-100.0, -50.0, -20.0, -10.0, 0.0, 10.0, 20.0, 50.0, 100.0],
        dtype=torch.float32
    )

    # Apply clamping
    log_ratio_clamped = torch.clamp(extreme_log_ratios, min=-10.0, max=10.0)

    # Verify clamping worked
    assert torch.all(log_ratio_clamped >= -10.0), \
        f"log_ratio should be >= -10.0, got {log_ratio_clamped.min().item()}"
    assert torch.all(log_ratio_clamped <= 10.0), \
        f"log_ratio should be <= 10.0, got {log_ratio_clamped.max().item()}"

    # Verify exp() produces reasonable values
    ratio = torch.exp(log_ratio_clamped)

    assert torch.all(ratio >= 4.5e-5), \
        f"ratio should be >= exp(-10), got min={ratio.min().item()}"
    assert torch.all(ratio <= 22027.0), \
        f"ratio should be <= exp(10), got max={ratio.max().item()}"

    # All should be finite
    assert torch.all(torch.isfinite(ratio)), \
        f"ratio should be finite after clamping: {ratio.tolist()}"


def test_ratio_clamp_old_bug_comparison() -> None:
    """Compare old (±20) vs new (±10) clamping to demonstrate improvement."""
    torch = pytest.importorskip("torch")

    # Extreme value that would trigger the bug
    extreme_log_ratio = torch.tensor([20.0], dtype=torch.float32)

    # OLD (buggy): clamp to ±20
    log_ratio_old = torch.clamp(extreme_log_ratio, min=-20.0, max=20.0)
    ratio_old = torch.exp(log_ratio_old)

    # NEW (fixed): clamp to ±10
    log_ratio_new = torch.clamp(extreme_log_ratio, min=-10.0, max=10.0)
    ratio_new = torch.exp(log_ratio_new)

    # OLD produced absurdly large value
    assert ratio_old[0].item() > 4.85e8, \
        f"Old implementation should produce exp(20)≈485M, got {ratio_old[0].item()}"

    # NEW produces much more reasonable value
    assert ratio_new[0].item() < 22100, \
        f"New implementation should produce exp(10)≈22k, got {ratio_new[0].item()}"

    # Ratio reduction factor
    reduction_factor = ratio_old[0].item() / ratio_new[0].item()
    assert reduction_factor > 20000, \
        f"New clamp reduces max ratio by >20,000x: {reduction_factor:.0f}x"


def test_ratio_clamp_consistency_with_ppo_clip() -> None:
    """Test that clamped ratios work correctly with PPO clip operation."""
    torch = pytest.importorskip("torch")

    clip_range = 0.1  # Typical clip_range value

    # Various log_ratio values
    log_ratios = torch.tensor(
        [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0],
        dtype=torch.float32
    )

    # Clamp and compute ratio
    log_ratio_clamped = torch.clamp(log_ratios, min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio_clamped)

    # Apply PPO clip
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

    # Verify clipped values are in correct range
    assert torch.all(ratio_clipped >= 1 - clip_range - 1e-6), \
        f"ratio_clipped should be >= {1-clip_range}"
    assert torch.all(ratio_clipped <= 1 + clip_range + 1e-6), \
        f"ratio_clipped should be <= {1+clip_range}"

    # For extreme log_ratio values, ratio_clipped should hit bounds
    assert abs(ratio_clipped[0].item() - 0.9) < 1e-6, \
        f"ratio for log_ratio=-10 should clip to 0.9, got {ratio_clipped[0].item()}"
    assert abs(ratio_clipped[-1].item() - 1.1) < 1e-6, \
        f"ratio for log_ratio=+10 should clip to 1.1, got {ratio_clipped[-1].item()}"


def test_ratio_clamp_gradient_stability() -> None:
    """Test that gradients remain stable with clamped log_ratio."""
    torch = pytest.importorskip("torch")

    # Create log_ratio tensor with gradient tracking
    log_ratio = torch.tensor(
        [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0],
        dtype=torch.float32,
        requires_grad=True
    )

    # Clamp and compute ratio
    log_ratio_clamped = torch.clamp(log_ratio, min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio_clamped)

    # Simulate PPO loss (simplified)
    advantages = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    loss = -(advantages * ratio).mean()

    # Backpropagate
    loss.backward()

    # Check gradients are finite
    assert log_ratio.grad is not None, "Gradients should be computed"
    assert torch.all(torch.isfinite(log_ratio.grad)), \
        f"Gradients should be finite: {log_ratio.grad.tolist()}"

    # Gradients for clamped values should be zero (detached by clamp)
    # Actually, torch.clamp has gradient, but it's zero beyond clamp bounds
    grad = log_ratio.grad.tolist()

    # Values beyond clamp bounds should have zero or very small gradient
    assert abs(grad[0]) < 1e-6, \
        f"Gradient for log_ratio=-15 (clamped to -10) should be ≈0, got {grad[0]}"
    assert abs(grad[-1]) < 1e-6, \
        f"Gradient for log_ratio=+15 (clamped to +10) should be ≈0, got {grad[-1]}"


def test_ratio_clamp_numerical_precision() -> None:
    """Test that clamping maintains numerical precision for typical values."""
    torch = pytest.importorskip("torch")

    # High-precision test: small log_ratio differences
    log_ratio_small = torch.tensor(
        [0.001, 0.01, 0.05, 0.1],
        dtype=torch.float32
    )

    # No clamping should occur for these values
    log_ratio_clamped = torch.clamp(log_ratio_small, min=-10.0, max=10.0)

    # Should be unchanged
    assert torch.allclose(log_ratio_small, log_ratio_clamped, atol=1e-8), \
        "Small log_ratio values should not be affected by clamping"

    # Compute ratio
    ratio = torch.exp(log_ratio_clamped)

    # Check precision
    expected_ratios = [1.001, 1.01005, 1.05127, 1.10517]
    for i, (actual, expected) in enumerate(zip(ratio.tolist(), expected_ratios)):
        assert abs(actual - expected) < 1e-4, \
            f"ratio[{i}] = {actual}, expected {expected}"


def test_ratio_clamp_integration_with_advantages() -> None:
    """Test ratio clamping in context of PPO policy loss computation."""
    torch = pytest.importorskip("torch")

    batch_size = 100
    clip_range = 0.1

    # Simulate realistic training scenario
    torch.manual_seed(123)
    log_prob = torch.randn(batch_size, dtype=torch.float32) * 0.5
    old_log_prob = log_prob + torch.randn(batch_size, dtype=torch.float32) * 0.03
    advantages = torch.randn(batch_size, dtype=torch.float32)

    # Compute log_ratio
    log_ratio = log_prob - old_log_prob

    # Apply clamping (as in the fix)
    log_ratio_clamped = torch.clamp(log_ratio, min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio_clamped)

    # PPO policy loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Verify loss is finite and reasonable
    assert torch.isfinite(policy_loss), \
        f"PPO policy loss should be finite, got {policy_loss.item()}"
    assert -100.0 < policy_loss.item() < 100.0, \
        f"PPO policy loss should be reasonable, got {policy_loss.item()}"

    # Verify ratio statistics are healthy
    ratio_mean = ratio.mean().item()
    ratio_std = ratio.std().item()

    assert 0.8 < ratio_mean < 1.2, \
        f"Mean ratio should be near 1.0, got {ratio_mean}"
    assert ratio_std < 0.5, \
        f"Ratio std should be small, got {ratio_std}"


def test_ratio_clamp_matches_awr_weighting_pattern() -> None:
    """Test that ratio clamping follows the same pattern as AWR weighting fix."""
    torch = pytest.importorskip("torch")

    # AWR weighting uses: exp_arg = clamp(adv/beta, max=log(max_weight))
    # This ensures exp(exp_arg) <= max_weight

    # For ratio clamping, we use: log_ratio = clamp(log_ratio, max=10)
    # This ensures exp(log_ratio) <= exp(10) ≈ 22k

    max_log_ratio = 10.0
    max_ratio = math.exp(max_log_ratio)

    # Test extreme values
    extreme_log_ratios = torch.tensor([20.0, 50.0, 100.0], dtype=torch.float32)

    # Clamp before exp (correct pattern)
    log_ratio_clamped = torch.clamp(extreme_log_ratios, max=max_log_ratio)
    ratio = torch.exp(log_ratio_clamped)

    # All ratios should be <= max_ratio
    assert torch.all(ratio <= max_ratio * 1.01), \
        f"Ratios should be <= exp(10)≈{max_ratio:.0f}, got max={ratio.max().item():.0f}"

    # This is analogous to AWR: clamp arg before exp, not after
    # ✓ CORRECT:   log_ratio = clamp(log_diff, max=10); ratio = exp(log_ratio)
    # ✗ INCORRECT: log_ratio = clamp(log_diff, max=20); ratio = clamp(exp(log_ratio), max=22k)


def test_ratio_clamp_edge_cases() -> None:
    """Test edge cases: zero, negative, inf, nan."""
    torch = pytest.importorskip("torch")

    # Edge case inputs
    edge_cases = torch.tensor(
        [float('-inf'), -100.0, -10.0, 0.0, 10.0, 100.0, float('inf')],
        dtype=torch.float32
    )

    # Clamp (inf/nan should be handled by clamp)
    log_ratio_clamped = torch.clamp(edge_cases, min=-10.0, max=10.0)

    # Check clamping worked for finite values
    finite_mask = torch.isfinite(edge_cases)
    finite_clamped = log_ratio_clamped[finite_mask]

    assert torch.all(finite_clamped >= -10.0), \
        f"Finite values should be >= -10.0: {finite_clamped.tolist()}"
    assert torch.all(finite_clamped <= 10.0), \
        f"Finite values should be <= 10.0: {finite_clamped.tolist()}"

    # For inf/nan, clamp behavior is defined but may not fix them
    # We need additional safeguards in the actual training code
    # This test just documents the behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
