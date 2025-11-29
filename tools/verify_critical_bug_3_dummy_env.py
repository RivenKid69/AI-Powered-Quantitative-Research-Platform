"""
Critical Bug #3 Verification: SimpleDummyEnv Invalid Type

Hypothesis: SimpleDummyEnv in test_twin_critics_integration.py does not inherit
from gymnasium.Env, causing ValueError when trying to create VecEnv.

Expected: ValueError about environment type
Location: tests/test_twin_critics_integration.py:20
"""

import sys
import numpy as np
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class SimpleDummyEnv:
    """Reproduction of SimpleDummyEnv from test_twin_critics_integration.py"""

    def __init__(self):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.steps = 0
        obs = np.random.randn(10).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1
        obs = np.random.randn(10).astype(np.float32)
        reward = -np.sum(action**2)
        done = self.steps >= self.max_steps
        truncated = False
        return obs, reward, done, truncated, {}


class ProperDummyEnv(gymnasium.Env):
    """Properly inheriting from gymnasium.Env"""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.steps = 0
        obs = np.random.randn(10).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1
        obs = np.random.randn(10).astype(np.float32)
        reward = -np.sum(action**2)
        done = self.steps >= self.max_steps
        truncated = False
        return obs, reward, done, truncated, {}


def test_simple_dummy_env():
    """Test if SimpleDummyEnv causes ValueError."""

    print("=" * 80)
    print("CRITICAL BUG #3: SimpleDummyEnv Invalid Type")
    print("=" * 80)
    print()

    # Test 1: SimpleDummyEnv (problematic)
    print("Test 1: SimpleDummyEnv (without gymnasium.Env inheritance)")
    print("-" * 40)

    try:
        env = DummyVecEnv([lambda: SimpleDummyEnv()])
        print("  ✓ DummyVecEnv created successfully")

        # Try to reset
        obs = env.reset()
        print(f"  ✓ Environment reset successful, obs shape: {obs.shape}")

        # Try a step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"  ✓ Environment step successful")

        env.close()
        print()
        print("Result: SimpleDummyEnv works fine")
        bug_1 = False

    except ValueError as e:
        print(f"  ✗ ValueError: {e}")
        print()
        print("Result: BUG CONFIRMED - SimpleDummyEnv causes ValueError")
        bug_1 = True

    except Exception as e:
        print(f"  ✗ Unexpected error: {type(e).__name__}: {e}")
        bug_1 = True

    print()

    # Test 2: ProperDummyEnv (should work)
    print("Test 2: ProperDummyEnv (with gymnasium.Env inheritance)")
    print("-" * 40)

    try:
        env = DummyVecEnv([lambda: ProperDummyEnv()])
        print("  ✓ DummyVecEnv created successfully")

        obs = env.reset()
        print(f"  ✓ Environment reset successful, obs shape: {obs.shape}")

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"  ✓ Environment step successful")

        env.close()
        print()
        print("Result: ProperDummyEnv works correctly")

    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")
        print("  (This should not happen!)")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Check inheritance
    print("Inheritance check:")
    print(f"  SimpleDummyEnv is gymnasium.Env: {isinstance(SimpleDummyEnv(), gymnasium.Env)}")
    print(f"  ProperDummyEnv is gymnasium.Env:  {isinstance(ProperDummyEnv(), gymnasium.Env)}")
    print()

    if bug_1:
        print("VERDICT: BUG EXISTS ✗")
        print("Severity: MEDIUM - Test code issue, not production code")
        print("Impact: Breaks test_twin_critics_integration.py tests")
        print()
        print("Fix: SimpleDummyEnv should inherit from gymnasium.Env")
        return True
    else:
        print("VERDICT: BUG NOT FOUND ✓")
        print("SimpleDummyEnv works fine (or SB3 is more forgiving)")
        return False


if __name__ == "__main__":
    print("\n")
    bug_exists = test_simple_dummy_env()
    print("\n")
    sys.exit(1 if bug_exists else 0)
