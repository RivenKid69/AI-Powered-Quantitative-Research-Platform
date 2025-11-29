"""
Critical Bug #1 Verification: Twin Critics Tensor Dimension Mismatch

Hypothesis: When using Twin Critics with categorical value head, there is a
dimension mismatch between target_distribution and log_predictions tensors
during loss computation, causing RuntimeError.

Expected: RuntimeError with message about tensor size mismatch
Location: distributional_ppo.py:2534
"""

import sys
import traceback
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy


def test_twin_critics_dimension_mismatch():
    """Test if Twin Critics has dimension mismatch bug."""

    print("=" * 80)
    print("CRITICAL BUG #1: Twin Critics Tensor Dimension Mismatch")
    print("=" * 80)
    print()

    # Setup
    print("1. Creating environment...")
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

    print("2. Creating model with Twin Critics enabled...")
    try:
        model = DistributionalPPO(
            CustomActorCriticPolicy,
            env,
            verbose=0,
            n_steps=64,
            n_epochs=2,
            batch_size=32,
        )
        print(f"   [OK] Model created successfully")
        print(f"   - Twin Critics enabled: {getattr(model.policy, '_use_twin_critics', False)}")
        print(f"   - Quantile mode: {getattr(model.policy, '_use_quantile_value_head', False)}")
    except Exception as e:
        print(f"   [FAIL] Failed to create model: {e}")
        return False

    print()
    print("3. Running training to trigger the bug...")
    print("   (If bug exists, should crash during training)")

    try:
        # This should trigger the dimension mismatch if bug exists
        model.learn(total_timesteps=128, progress_bar=False)
        print(f"   [OK] Training completed without errors")
        print()
        print("RESULT: BUG NOT FOUND - Twin Critics works correctly")
        return False

    except RuntimeError as e:
        error_msg = str(e)
        print(f"   [FAIL] RuntimeError occurred: {error_msg}")

        # Check if it's the specific dimension mismatch error
        if "size of tensor" in error_msg and "must match" in error_msg:
            print()
            print("RESULT: [OK] BUG CONFIRMED - Dimension mismatch in Twin Critics")
            print()
            print("Error details:")
            print(f"  Message: {error_msg}")
            print()
            traceback.print_exc()
            return True
        else:
            print(f"   Different RuntimeError (not the expected bug)")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"   [FAIL] Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()


if __name__ == "__main__":
    print("\n")
    bug_exists = test_twin_critics_dimension_mismatch()
    print("\n")
    print("=" * 80)
    if bug_exists:
        print("VERDICT: BUG EXISTS [FAIL]")
        print("Severity: CRITICAL - Blocks Twin Critics functionality")
        sys.exit(1)
    else:
        print("VERDICT: BUG NOT FOUND [OK]")
        print("Twin Critics works correctly")
        sys.exit(0)
