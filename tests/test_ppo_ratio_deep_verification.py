"""
DEEP VERIFICATION TESTS: PPO log_ratio handling with extreme edge cases.

These tests go BEYOND standard testing to verify:
1. Numerical overflow behavior (log_ratio > 88 → exp = inf)
2. Loss computation with inf/nan ratios
3. Gradient flow with extreme values
4. Interaction between inf ratio and PPO clipping
5. Backward pass stability

The goal is to find ANY potential issues with removing log_ratio clamping.
"""

import math
import pytest
import torch
import numpy as np


def test_extreme_log_ratio_causes_overflow():
    """CRITICAL: Test that very large log_ratio causes exp overflow to inf."""
    torch = pytest.importorskip("torch")

    # Float32 exp overflow threshold is ~88
    test_cases = [
        (50.0, "should_be_finite"),
        (88.0, "borderline"),
        (89.0, "should_overflow"),
        (100.0, "definitely_overflow"),
        (1000.0, "extreme_overflow"),
    ]

    for log_ratio_val, description in test_cases:
        log_ratio = torch.tensor([log_ratio_val], dtype=torch.float32)
        ratio = torch.exp(log_ratio)

        print(f"log_ratio={log_ratio_val}: ratio={ratio.item()}, "
              f"finite={torch.isfinite(ratio).item()}, desc={description}")

        # Document the behavior
        if log_ratio_val >= 89:
            assert torch.isinf(ratio), \
                f"exp({log_ratio_val}) should be inf, got {ratio.item()}"


def test_inf_ratio_with_ppo_loss():
    """CRITICAL: Test PPO loss computation when ratio = inf."""
    torch = pytest.importorskip("torch")

    # Create scenario where ratio = inf
    log_ratio = torch.tensor([100.0, 0.0, -100.0], dtype=torch.float32)
    ratio = torch.exp(log_ratio)  # [inf, 1.0, ~0]

    advantages = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    clip_range = 0.1

    # PPO loss computation
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    print(f"ratio: {ratio.tolist()}")
    print(f"policy_loss_1: {policy_loss_1.tolist()}")
    print(f"policy_loss_2: {policy_loss_2.tolist()}")
    print(f"policy_loss: {policy_loss.item()}")

    # Check if loss is finite
    if not torch.isfinite(policy_loss):
        print("⚠️  PROBLEM FOUND: Loss is not finite!")
        print("This will cause NaN gradients and training failure!")
        # This is a CRITICAL issue that needs to be addressed
        assert False, "Loss must be finite for stable training"


def test_negative_advantage_with_inf_ratio():
    """CRITICAL: Test the most dangerous case: negative advantage + inf ratio."""
    torch = pytest.importorskip("torch")

    # Worst case: ratio = inf, advantage < 0
    log_ratio = torch.tensor([100.0], dtype=torch.float32)
    ratio = torch.exp(log_ratio)  # inf
    advantage = torch.tensor([-1.0], dtype=torch.float32)
    clip_range = 0.1

    # PPO loss computation
    policy_loss_1 = advantage * ratio  # -1.0 * inf = -inf
    policy_loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)  # -1.0 * 1.1

    print(f"ratio: {ratio.item()}")
    print(f"advantage: {advantage.item()}")
    print(f"policy_loss_1: {policy_loss_1.item()}")
    print(f"policy_loss_2: {policy_loss_2.item()}")

    # min(-inf, finite) = -inf
    policy_loss_min = torch.min(policy_loss_1, policy_loss_2)
    print(f"min(policy_loss_1, policy_loss_2): {policy_loss_min.item()}")

    # Final loss: -min = -(-inf) = inf
    policy_loss = -policy_loss_min.mean()
    print(f"Final loss: {policy_loss.item()}")

    # This is CRITICAL FAILURE
    if torch.isinf(policy_loss):
        print("🚨 CRITICAL BUG FOUND!")
        print("When ratio=inf and advantage<0, loss becomes inf!")
        print("This will propagate NaN gradients and break training!")
        assert False, "This is a critical numerical stability issue"


def test_gradient_flow_with_inf_ratio():
    """CRITICAL: Test gradient computation when ratio = inf."""
    torch = pytest.importorskip("torch")

    # Create differentiable log_prob that will produce inf ratio
    log_prob = torch.tensor([100.0], dtype=torch.float32, requires_grad=True)
    old_log_prob = torch.tensor([0.0], dtype=torch.float32)
    advantage = torch.tensor([-1.0], dtype=torch.float32)

    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)  # inf

    clip_range = 0.1
    policy_loss_1 = advantage * ratio
    policy_loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    print(f"Loss before backward: {loss.item()}")

    try:
        loss.backward()
        print(f"Gradient: {log_prob.grad}")

        if log_prob.grad is not None:
            if torch.isnan(log_prob.grad):
                print("🚨 GRADIENT IS NaN!")
                assert False, "NaN gradient will break training"
            elif torch.isinf(log_prob.grad):
                print("🚨 GRADIENT IS INF!")
                assert False, "Inf gradient will break training"
    except RuntimeError as e:
        print(f"🚨 BACKWARD FAILED: {e}")
        assert False, f"Backward pass failed: {e}"


def test_mixed_inf_finite_ratio_batch():
    """Test batch with mix of inf and finite ratios."""
    torch = pytest.importorskip("torch")

    # Batch with normal, large, and overflow values
    log_ratios = torch.tensor([0.1, 1.0, 10.0, 50.0, 100.0], dtype=torch.float32)
    ratios = torch.exp(log_ratios)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5, 1.0], dtype=torch.float32)

    clip_range = 0.1

    policy_loss_1 = advantages * ratios
    policy_loss_2 = advantages * torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    print(f"ratios: {ratios.tolist()}")
    print(f"policy_loss_1: {policy_loss_1.tolist()}")
    print(f"policy_loss_2: {policy_loss_2.tolist()}")
    print(f"Final loss: {policy_loss.item()}")

    if not torch.isfinite(policy_loss):
        print("🚨 PROBLEM: Loss is not finite with mixed batch!")
        assert False, "Loss must remain finite even with some inf ratios"


def test_actual_distributional_ppo_code_path():
    """Test the EXACT code path from distributional_ppo.py with extreme values."""
    torch = pytest.importorskip("torch")

    # Simulate the exact scenario from distributional_ppo.py:7874-7880
    batch_size = 100

    # Create realistic batch with one extreme outlier
    torch.manual_seed(42)
    log_prob_selected = torch.randn(batch_size, dtype=torch.float32) * 0.5
    old_log_prob_selected = log_prob_selected.clone()

    # Inject extreme value (simulating policy collapse or numerical instability)
    log_prob_selected[0] = 100.0  # This will cause exp overflow
    old_log_prob_selected[0] = 0.0

    advantages_selected = torch.randn(batch_size, dtype=torch.float32)
    advantages_selected[0] = -1.0  # Worst case: negative advantage

    clip_range = 0.05

    # EXACT code from distributional_ppo.py
    log_ratio = log_prob_selected - old_log_prob_selected
    ratio = torch.exp(log_ratio)
    policy_loss_1 = advantages_selected * ratio
    policy_loss_2 = advantages_selected * torch.clamp(
        ratio, 1 - clip_range, 1 + clip_range
    )
    policy_loss_ppo = -torch.min(policy_loss_1, policy_loss_2).mean()

    print(f"\nExact distributional_ppo.py code path:")
    print(f"log_ratio[0]: {log_ratio[0].item()}")
    print(f"ratio[0]: {ratio[0].item()}")
    print(f"policy_loss_1[0]: {policy_loss_1[0].item()}")
    print(f"policy_loss_2[0]: {policy_loss_2[0].item()}")
    print(f"policy_loss_ppo: {policy_loss_ppo.item()}")

    # Check if loss is finite
    if not torch.isfinite(policy_loss_ppo):
        print("\n🚨 CRITICAL BUG CONFIRMED IN ACTUAL CODE PATH!")
        print("The exact code from distributional_ppo.py produces non-finite loss!")
        print("This WILL break training when extreme log_prob differences occur!")
        assert False, "CRITICAL: distributional_ppo.py code produces non-finite loss"


def test_comparison_old_clamp_vs_no_clamp():
    """Compare numerical stability: old clamp(±10) vs no clamp."""
    torch = pytest.importorskip("torch")

    # Test case: log_ratio = 100
    log_ratio = torch.tensor([100.0], dtype=torch.float32)
    advantage = torch.tensor([-1.0], dtype=torch.float32)
    clip_range = 0.1

    # OLD: with clamp ±10
    log_ratio_clamped = torch.clamp(log_ratio, min=-10.0, max=10.0)
    ratio_old = torch.exp(log_ratio_clamped)
    loss_1_old = advantage * ratio_old
    loss_2_old = advantage * torch.clamp(ratio_old, 1 - clip_range, 1 + clip_range)
    loss_old = -torch.min(loss_1_old, loss_2_old).mean()

    # NEW: no clamp
    ratio_new = torch.exp(log_ratio)
    loss_1_new = advantage * ratio_new
    loss_2_new = advantage * torch.clamp(ratio_new, 1 - clip_range, 1 + clip_range)
    loss_new = -torch.min(loss_1_new, loss_2_new).mean()

    print("\n=== OLD (clamp ±10) ===")
    print(f"ratio: {ratio_old.item()}, finite: {torch.isfinite(ratio_old).item()}")
    print(f"loss: {loss_old.item()}, finite: {torch.isfinite(loss_old).item()}")

    print("\n=== NEW (no clamp) ===")
    print(f"ratio: {ratio_new.item()}, finite: {torch.isfinite(ratio_new).item()}")
    print(f"loss: {loss_new.item()}, finite: {torch.isfinite(loss_new).item()}")

    # OLD was actually PROTECTING us!
    if torch.isfinite(loss_old) and not torch.isfinite(loss_new):
        print("\n⚠️  OLD IMPLEMENTATION WAS PROTECTING FROM OVERFLOW!")
        print("Removing clamp completely causes numerical instability!")
        print("We need a better solution!")


def test_what_is_safe_clamp_range():
    """Determine what clamp range provides both theory and numerical stability."""
    torch = pytest.importorskip("torch")

    # Test various clamp ranges
    clamp_ranges = [10.0, 20.0, 30.0, 40.0, 50.0, 85.0, 88.0, None]

    advantage = torch.tensor([-1.0], dtype=torch.float32)
    clip_range = 0.1

    print("\n=== Testing different clamp ranges ===")
    for clamp_max in clamp_ranges:
        log_ratio = torch.tensor([100.0], dtype=torch.float32)

        if clamp_max is not None:
            log_ratio_clamped = torch.clamp(log_ratio, min=-clamp_max, max=clamp_max)
        else:
            log_ratio_clamped = log_ratio

        ratio = torch.exp(log_ratio_clamped)
        loss_1 = advantage * ratio
        loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        loss = -torch.min(loss_1, loss_2).mean()

        finite = torch.isfinite(loss).item()
        desc = "✅ STABLE" if finite else "❌ UNSTABLE"

        clamp_str = f"±{clamp_max}" if clamp_max else "None"
        print(f"clamp={clamp_str:8s}: ratio={ratio.item():.2e}, "
              f"loss_finite={finite}, {desc}")


def test_proper_solution_with_safety_clamp():
    """Test PROPER solution: wide safety clamp that doesn't affect normal training."""
    torch = pytest.importorskip("torch")

    # Use clamp at float32 limit: ±85 (exp(85) ≈ 2.6e36, still finite)
    safety_clamp = 85.0

    # Test 1: Normal training values - clamp should NOT activate
    log_ratio_normal = torch.randn(1000, dtype=torch.float32) * 0.05
    log_ratio_clamped = torch.clamp(log_ratio_normal, min=-safety_clamp, max=safety_clamp)

    assert torch.allclose(log_ratio_normal, log_ratio_clamped, rtol=1e-6), \
        "Safety clamp should not affect normal values"

    # Test 2: Extreme values - clamp should activate and prevent overflow
    log_ratio_extreme = torch.tensor([100.0, -100.0], dtype=torch.float32)
    log_ratio_clamped = torch.clamp(log_ratio_extreme, min=-safety_clamp, max=safety_clamp)
    ratio = torch.exp(log_ratio_clamped)

    assert torch.all(torch.isfinite(ratio)), \
        "Safety clamp should prevent overflow"

    # Test 3: PPO loss should be stable
    advantage = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    clip_range = 0.1

    loss_1 = advantage * ratio
    loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    loss = -torch.min(loss_1, loss_2).mean()

    assert torch.isfinite(loss), \
        "Loss should be finite with safety clamp"

    print("\n✅ PROPER SOLUTION VERIFIED:")
    print(f"   Use clamp(log_ratio, min=-85, max=85)")
    print(f"   - Does NOT affect normal training (log_ratio typically << 85)")
    print(f"   - PREVENTS overflow (exp(85) is still finite)")
    print(f"   - Maintains theoretical correctness (clamp is so wide it rarely activates)")
    print(f"   - Provides numerical safety against extreme edge cases")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
