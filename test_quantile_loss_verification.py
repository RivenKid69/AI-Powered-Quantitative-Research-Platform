"""
Test to verify quantile loss implementation correctness.

Reference: Dabney et al. 2018 "Distributional Reinforcement Learning with Quantile Regression"
Standard quantile loss formula: ρ_τ(u) = u * (τ - I{u < 0})
where u = target - predicted
"""
import torch
import numpy as np


def standard_quantile_loss(predicted, target, tau):
    """
    Standard quantile regression loss from literature.

    Args:
        predicted: predicted quantile value
        target: target value
        tau: quantile level (e.g., 0.25 for 25th percentile)

    Returns:
        loss value
    """
    # u = target - predicted (standard definition)
    u = target - predicted
    indicator = (u < 0).float()
    # ρ_τ(u) = u * (τ - I{u < 0})
    loss = u * (tau - indicator)
    return loss.abs().mean()  # Use absolute value for Huber-like behavior


def current_implementation_quantile_loss(predicted, target, tau):
    """
    Current implementation in distributional_ppo.py (simplified, without Huber).

    Line 2544-2553:
        delta = predicted_quantiles - targets  # Q - T
        indicator = (delta.detach() < 0.0).float()  # I{Q < T}
        loss_per_quantile = torch.abs(tau - indicator) * delta.abs()
    """
    # delta = predicted - target (inverted from standard!)
    delta = predicted - target
    indicator = (delta < 0.0).float()
    # Loss coefficient is |tau - I{delta < 0}|
    # When delta < 0 (Q < T): coefficient = |tau - 1| = 1 - tau
    # When delta >= 0 (Q >= T): coefficient = |tau - 0| = tau
    loss = torch.abs(tau - indicator) * delta.abs()
    return loss.mean()


def test_quantile_loss_behavior():
    """
    Test if the two implementations produce the same gradient direction.
    """
    print("=" * 80)
    print("Testing Quantile Loss Implementation")
    print("=" * 80)

    # Test case 1: Underestimation (Q < T)
    # For tau=0.75 quantile, if predicted < target, we should push predicted UP
    print("\nTest 1: Underestimation (predicted < target)")
    print("-" * 80)
    tau = 0.75
    predicted = torch.tensor(10.0, requires_grad=True)
    target = torch.tensor(20.0)

    # Standard loss
    loss_std = standard_quantile_loss(predicted, target, tau)
    loss_std.backward()
    grad_std = predicted.grad.item()
    predicted.grad = None

    # Current implementation
    loss_cur = current_implementation_quantile_loss(predicted, target, tau)
    loss_cur.backward()
    grad_cur = predicted.grad.item()

    print(f"Quantile level τ: {tau}")
    print(f"Predicted: {predicted.item()}, Target: {target.item()}")
    print(f"Standard loss: {loss_std.item():.4f}, gradient: {grad_std:.4f}")
    print(f"Current loss:  {loss_cur.item():.4f}, gradient: {grad_cur:.4f}")
    print(f"Gradient sign matches: {np.sign(grad_std) == np.sign(grad_cur)}")

    # Test case 2: Overestimation (Q > T)
    print("\nTest 2: Overestimation (predicted > target)")
    print("-" * 80)
    tau = 0.25
    predicted = torch.tensor(30.0, requires_grad=True)
    target = torch.tensor(20.0)

    # Standard loss
    loss_std = standard_quantile_loss(predicted, target, tau)
    loss_std.backward()
    grad_std = predicted.grad.item()
    predicted.grad = None

    # Current implementation
    loss_cur = current_implementation_quantile_loss(predicted, target, tau)
    loss_cur.backward()
    grad_cur = predicted.grad.item()

    print(f"Quantile level τ: {tau}")
    print(f"Predicted: {predicted.item()}, Target: {target.item()}")
    print(f"Standard loss: {loss_std.item():.4f}, gradient: {grad_std:.4f}")
    print(f"Current loss:  {loss_cur.item():.4f}, gradient: {grad_cur:.4f}")
    print(f"Gradient sign matches: {np.sign(grad_std) == np.sign(grad_cur)}")

    # Test case 3: Multiple quantiles
    print("\nTest 3: Multiple quantiles simultaneously")
    print("-" * 80)
    taus = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    predicted = torch.tensor([15.0, 18.0, 20.0, 22.0, 25.0], requires_grad=True)
    target = torch.tensor([20.0, 20.0, 20.0, 20.0, 20.0])

    print(f"Quantile levels τ: {taus.tolist()}")
    print(f"Predicted: {predicted.detach().tolist()}")
    print(f"Target: {target.tolist()}")

    # Standard loss
    losses_std = []
    for i, tau in enumerate(taus):
        loss = standard_quantile_loss(predicted[i], target[i], tau)
        losses_std.append(loss.item())

    # Current implementation
    losses_cur = []
    for i, tau in enumerate(taus):
        loss = current_implementation_quantile_loss(predicted[i], target[i], tau)
        losses_cur.append(loss.item())

    print(f"Standard losses: {[f'{l:.4f}' for l in losses_std]}")
    print(f"Current losses:  {[f'{l:.4f}' for l in losses_cur]}")

    print("\n" + "=" * 80)
    print("Theoretical Analysis:")
    print("=" * 80)
    print("\nStandard quantile loss (from literature):")
    print("  ρ_τ(u) = u * (τ - I{u < 0}), where u = target - predicted")
    print("  When Q < T (underestimation, u > 0): loss = u * τ")
    print("  When Q > T (overestimation, u < 0): loss = -u * (1 - τ) = |u| * (1 - τ)")
    print("\nCurrent implementation:")
    print("  delta = predicted - target (inverted!)")
    print("  loss = |τ - I{delta < 0}| * |delta|")
    print("  When Q < T (delta < 0): loss = |τ - 1| * |delta| = (1 - τ) * |delta|")
    print("  When Q > T (delta > 0): loss = |τ - 0| * |delta| = τ * |delta|")
    print("\nCoefficients are INVERTED!")
    print("  Standard: underestimation → τ, overestimation → (1-τ)")
    print("  Current:  underestimation → (1-τ), overestimation → τ")


if __name__ == "__main__":
    test_quantile_loss_behavior()
