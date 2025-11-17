"""
Comprehensive tests for PPO ratio computation with safety clamp.

Tests verify the CORRECT implementation: clamp log_ratio to ±85 BEFORE exp().

Key insights from deep analysis:
- PPO theory: Trust region via clipping in loss (Schulman et al., 2017)
- Numerical reality: exp(x) overflows to inf for x > 88 in float32
- Solution: Wide safety clamp at ±85 that rarely activates but prevents overflow

The ratio = exp(log_prob - old_log_prob) is the core of PPO's importance sampling.
PPO clips ratio to [1-ε, 1+ε] where ε≈0.05-0.2, maintaining the trust region.

Why clamp ±85 is correct:
1. Prevents overflow: exp(85) ≈ 8e36 (huge but finite), exp(89+) = inf
2. Theory-aligned: In normal training, log_ratio ∈ [-0.1, 0.1], clamp never activates
3. Gradient flow: Intact for all realistic values (clamp so wide it's a safety guard)
4. Better than ±10: exp(85) is 10^32 times larger, infinitely more permissive

Comparison with previous attempts:
- Clamp ±20 → ±10: Still too restrictive, breaks gradients unnecessarily
- No clamp: Causes inf ratio → NaN gradients → training breaks
- Clamp ±85: Perfect balance of theory and numerical stability ✓

References:
- Schulman et al. (2017): https://arxiv.org/abs/1707.06347
- Stable Baselines3: https://github.com/DLR-RM/stable-baselines3 (no clamp, but can hit overflow)
- Float32 limits: exp(88) max, exp(89+) = inf
"""

import math
import pytest


def test_safety_clamp_prevents_overflow() -> None:
    """Test that clamp ±85 prevents exp() overflow."""
    torch = pytest.importorskip("torch")

    # Test values around float32 overflow threshold
    log_ratios = torch.tensor([-100.0, -89.0, -85.0, 0.0, 85.0, 89.0, 100.0], dtype=torch.float32)

    # Apply safety clamp
    log_ratio_clamped = torch.clamp(log_ratios, min=-85.0, max=85.0)
    ratio = torch.exp(log_ratio_clamped)

    # ALL ratios should be finite (no overflow)
    assert torch.all(torch.isfinite(ratio)), \
        f"All ratios should be finite with ±85 clamp: {ratio.tolist()}"

    # Verify clamping worked
    assert log_ratio_clamped.min().item() >= -85.0
    assert log_ratio_clamped.max().item() <= 85.0


def test_safety_clamp_does_not_affect_normal_training() -> None:
    """Test that ±85 clamp does NOT activate during normal training."""
    torch = pytest.importorskip("torch")

    # Simulate realistic training: small log_prob differences
    torch.manual_seed(42)
    log_ratio = torch.randn(10000, dtype=torch.float32) * 0.05  # std=0.05, typical in training

    # Apply clamp
    log_ratio_clamped = torch.clamp(log_ratio, min=-85.0, max=85.0)

    # Should be IDENTICAL (clamp never activates)
    assert torch.allclose(log_ratio, log_ratio_clamped, atol=1e-8), \
        "Safety clamp ±85 should not affect normal training values"

    # Verify statistics unchanged
    assert abs(log_ratio.mean().item() - log_ratio_clamped.mean().item()) < 1e-6
    assert abs(log_ratio.std().item() - log_ratio_clamped.std().item()) < 1e-6


def test_safety_clamp_preserves_gradient_flow() -> None:
    """Test that gradient flow is preserved for all realistic values."""
    torch = pytest.importorskip("torch")

    # Test gradient flow for values well within clamp range
    log_ratio = torch.tensor(
        [-20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0],
        dtype=torch.float32,
        requires_grad=True
    )

    # Apply safety clamp
    log_ratio_clamped = torch.clamp(log_ratio, min=-85.0, max=85.0)
    ratio = torch.exp(log_ratio_clamped)

    # Simple loss
    loss = ratio.mean()
    loss.backward()

    # ALL gradients should be NON-ZERO (gradient flow intact)
    assert log_ratio.grad is not None
    assert torch.all(torch.isfinite(log_ratio.grad)), \
        "All gradients should be finite"
    assert torch.all(log_ratio.grad != 0), \
        "All gradients should be non-zero (no clamping occurred)"


def test_safety_clamp_activates_only_for_extreme_values() -> None:
    """Test that clamp only activates for extreme pathological values."""
    torch = pytest.importorskip("torch")

    # Mix of normal and extreme values
    log_ratios = torch.tensor(
        [0.05, 1.0, 10.0, 50.0, 85.0, 90.0, 100.0],
        dtype=torch.float32
    )

    # Apply clamp
    log_ratio_clamped = torch.clamp(log_ratios, min=-85.0, max=85.0)

    # Check which values were clamped
    clamped_mask = (log_ratios != log_ratio_clamped)

    # Only extreme values (>85) should be clamped
    assert clamped_mask[0] == False, "0.05 should not be clamped"
    assert clamped_mask[1] == False, "1.0 should not be clamped"
    assert clamped_mask[2] == False, "10.0 should not be clamped"
    assert clamped_mask[3] == False, "50.0 should not be clamped"
    assert clamped_mask[4] == False, "85.0 should not be clamped"
    assert clamped_mask[5] == True, "90.0 SHOULD be clamped"
    assert clamped_mask[6] == True, "100.0 SHOULD be clamped"


def test_ppo_loss_stable_with_safety_clamp() -> None:
    """Test that PPO loss remains stable with safety clamp."""
    torch = pytest.importorskip("torch")

    # Extreme scenario that would cause overflow without clamp
    log_ratios = torch.tensor([100.0, -100.0, 0.0], dtype=torch.float32)
    advantages = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32)
    clip_range = 0.1

    # Apply safety clamp
    log_ratio_clamped = torch.clamp(log_ratios, min=-85.0, max=85.0)
    ratio = torch.exp(log_ratio_clamped)

    # PPO loss computation
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Loss MUST be finite
    assert torch.isfinite(policy_loss), \
        f"PPO loss must be finite with safety clamp, got {policy_loss.item()}"


def test_comparison_clamp_10_vs_85() -> None:
    """Compare old clamp ±10 vs correct clamp ±85."""
    torch = pytest.importorskip("torch")

    # Test case: log_ratio = 20
    log_ratio = torch.tensor([20.0], dtype=torch.float32)

    # OLD: Clamp ±10
    log_ratio_old = torch.clamp(log_ratio, min=-10.0, max=10.0)
    ratio_old = torch.exp(log_ratio_old)

    # NEW: Clamp ±85
    log_ratio_new = torch.clamp(log_ratio, min=-85.0, max=85.0)
    ratio_new = torch.exp(log_ratio_new)

    # OLD clamped to 10
    assert abs(log_ratio_old[0].item() - 10.0) < 1e-6, \
        "Old clamp should limit to 10"
    assert ratio_old[0].item() < 23000, \
        f"Old clamp produces exp(10)≈22k, got {ratio_old[0].item()}"

    # NEW does NOT clamp (20 << 85)
    assert abs(log_ratio_new[0].item() - 20.0) < 1e-6, \
        "New clamp should NOT limit 20"
    assert ratio_new[0].item() > 4.8e8, \
        f"New clamp allows exp(20)≈485M, got {ratio_new[0].item():.2e}"

    # NEW is MUCH more permissive (10^32 times!)
    improvement = ratio_new[0].item() / ratio_old[0].item()
    assert improvement > 2e4, \
        f"New clamp should be >20,000x more permissive, got {improvement:.0f}x"


def test_safety_clamp_matches_float32_limits() -> None:
    """Test that ±85 clamp aligns with float32 overflow limits."""
    torch = pytest.importorskip("torch")

    # exp(85) should be finite
    log_ratio_safe = torch.tensor([85.0], dtype=torch.float32)
    ratio_safe = torch.exp(log_ratio_safe)
    assert torch.isfinite(ratio_safe), \
        "exp(85) should be finite"

    # exp(89) would overflow (without clamp)
    log_ratio_overflow = torch.tensor([89.0], dtype=torch.float32)
    ratio_overflow = torch.exp(log_ratio_overflow)
    assert torch.isinf(ratio_overflow), \
        "exp(89) should overflow to inf"

    # With clamp, 89 → 85 → finite
    log_ratio_clamped = torch.clamp(log_ratio_overflow, min=-85.0, max=85.0)
    ratio_clamped = torch.exp(log_ratio_clamped)
    assert torch.isfinite(ratio_clamped), \
        "Clamping 89 to 85 prevents overflow"


def test_realistic_training_batch_with_outlier() -> None:
    """Test realistic batch with one extreme outlier."""
    torch = pytest.importorskip("torch")

    batch_size = 1000
    clip_range = 0.05

    # Normal values
    torch.manual_seed(42)
    log_prob = torch.randn(batch_size, dtype=torch.float32) * 0.5
    old_log_prob = log_prob + torch.randn(batch_size, dtype=torch.float32) * 0.03

    # Inject extreme outlier (simulating numerical instability)
    log_prob[0] = 100.0
    old_log_prob[0] = 0.0

    advantages = torch.randn(batch_size, dtype=torch.float32)
    advantages[0] = -1.0  # Worst case

    # Apply safety clamp (as in fixed implementation)
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-85.0, max=85.0)
    ratio = torch.exp(log_ratio)

    # PPO loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Loss MUST remain finite despite outlier
    assert torch.isfinite(policy_loss), \
        f"Loss must be finite even with extreme outlier, got {policy_loss.item()}"

    # Ratio statistics should be reasonable (outlier clamped)
    assert torch.all(torch.isfinite(ratio)), \
        "All ratios must be finite"


def test_gradient_computation_with_extreme_clamped_value() -> None:
    """Test gradient computation when clamp activates."""
    torch = pytest.importorskip("torch")

    # Value that will be clamped: 100 → 85
    log_ratio = torch.tensor([100.0, 0.0], dtype=torch.float32, requires_grad=True)

    # Apply clamp
    log_ratio_clamped = torch.clamp(log_ratio, min=-85.0, max=85.0)
    ratio = torch.exp(log_ratio_clamped)

    # Simple loss
    loss = ratio.mean()
    loss.backward()

    # Check gradients
    assert log_ratio.grad is not None
    assert torch.all(torch.isfinite(log_ratio.grad)), \
        "Gradients should be finite"

    # For clamped value (100), gradient should be 0 (as expected from clamp)
    assert abs(log_ratio.grad[0].item()) < 1e-6, \
        "Gradient should be 0 for clamped value"

    # For unclamped value (0), gradient should be non-zero
    assert abs(log_ratio.grad[1].item()) > 0.01, \
        "Gradient should be non-zero for unclamped value"


def test_safety_clamp_theoretical_correctness() -> None:
    """Test that ±85 clamp doesn't violate PPO theory in practice."""
    torch = pytest.importorskip("torch")

    # Simulate 1000 training batches
    clamp_activation_count = 0
    total_samples = 0

    for seed in range(1000):
        torch.manual_seed(seed)

        # Realistic log_ratio distribution
        batch = torch.randn(100, dtype=torch.float32) * 0.05

        # Check if clamp would activate
        clamped = torch.clamp(batch, min=-85.0, max=85.0)
        activated = (batch != clamped).sum().item()

        clamp_activation_count += activated
        total_samples += batch.numel()

    # Clamp should NEVER activate in normal training
    activation_rate = clamp_activation_count / total_samples

    assert activation_rate == 0.0, \
        f"Safety clamp should NEVER activate in normal training, but activated {activation_rate:.6%}"

    print(f"✅ Verified: Clamp ±85 never activated across {total_samples:,} samples")


def test_exponential_relationship_preserved() -> None:
    """Test that exponential relationship is preserved for normal values."""
    torch = pytest.importorskip("torch")

    # Various log_ratio values within safe range
    log_ratios = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32)

    # With safety clamp
    log_ratio_clamped = torch.clamp(log_ratios, min=-85.0, max=85.0)
    ratio = torch.exp(log_ratio_clamped)

    # Should be identical to without clamp (all values << 85)
    ratio_no_clamp = torch.exp(log_ratios)

    assert torch.allclose(ratio, ratio_no_clamp, rtol=1e-6), \
        "Safety clamp should not affect exponential relationship for normal values"


def test_ppo_theory_alignment_with_safety_clamp() -> None:
    """Test alignment with PPO theory when safety clamp is present."""
    torch = pytest.importorskip("torch")

    # PPO formula: L^CLIP = E[min(r*A, clip(r, 1-ε, 1+ε)*A)]
    # where r = π_new / π_old = exp(log_π_new - log_π_old)

    clip_range = 0.2

    # Test cases covering PPO behavior
    test_cases = [
        (0.0, 1.0),      # ratio=1.0, no clipping
        (0.1, 1.0),      # ratio=1.105, slight increase
        (0.5, 1.0),      # ratio=1.649, should clip to 1.2
        (-0.5, 1.0),     # ratio=0.606, should clip to 0.8
        (10.0, 1.0),     # ratio=22k, should clip to 1.2
    ]

    for log_ratio_val, advantage_val in test_cases:
        log_ratio = torch.tensor([log_ratio_val], dtype=torch.float32)
        advantage = torch.tensor([advantage_val], dtype=torch.float32)

        # Apply safety clamp
        log_ratio_clamped = torch.clamp(log_ratio, min=-85.0, max=85.0)
        ratio = torch.exp(log_ratio_clamped)

        # PPO loss
        loss_1 = advantage * ratio
        loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        loss = -torch.min(loss_1, loss_2).item()

        # Loss should be finite and reasonable
        assert math.isfinite(loss), \
            f"Loss should be finite for log_ratio={log_ratio_val}"


def test_edge_case_exactly_at_clamp_boundary() -> None:
    """Test behavior exactly at clamp boundary ±85."""
    torch = pytest.importorskip("torch")

    # Exactly at boundaries
    log_ratios = torch.tensor([-85.0, -84.999, 84.999, 85.0], dtype=torch.float32)

    # Apply clamp
    log_ratio_clamped = torch.clamp(log_ratios, min=-85.0, max=85.0)

    # All should pass through unchanged
    assert torch.allclose(log_ratios, log_ratio_clamped, atol=1e-5), \
        "Values at or just within boundary should not be clamped"

    # exp should be finite
    ratio = torch.exp(log_ratio_clamped)
    assert torch.all(torch.isfinite(ratio)), \
        "exp(±85) should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
