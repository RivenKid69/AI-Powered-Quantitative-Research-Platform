"""Quick test to verify Bug #3 fix in test_twin_critics_integration.py"""

import sys
sys.path.insert(0, 'C:\\Users\\suyun\\ai-quant-platform\\tests')

from test_twin_critics_integration import SimpleDummyEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium

print("=" * 80)
print("Testing Bug #3 Fix: SimpleDummyEnv inheritance")
print("=" * 80)
print()

print(f"SimpleDummyEnv inherits from gymnasium.Env: {issubclass(SimpleDummyEnv, gymnasium.Env)}")
print()

try:
    env = DummyVecEnv([lambda: SimpleDummyEnv()])
    print("[OK] DummyVecEnv created successfully")

    obs = env.reset()
    print(f"[OK] Environment reset successful, obs shape: {obs.shape}")

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"[OK] Environment step successful")

    env.close()
    print()
    print("=" * 80)
    print("VERDICT: BUG #3 FIXED [OK]")
    print("SimpleDummyEnv now properly inherits from gymnasium.Env")
    print("=" * 80)

except Exception as e:
    print(f"[FAIL] Error: {type(e).__name__}: {e}")
    print()
    print("=" * 80)
    print("VERDICT: BUG #3 NOT FIXED [FAIL]")
    print("=" * 80)
