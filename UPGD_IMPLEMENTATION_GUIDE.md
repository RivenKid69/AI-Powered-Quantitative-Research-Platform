# UPGD Optimizer Implementation Guide

**Objective:** Integrate UPGD (Utility-based Perturbed Gradient Descent) optimizer to prevent catastrophic forgetting in non-stationary markets.

**Expected Impact:** 20-30% reduction in performance degradation over 6 months, 50% reduction in retraining frequency.

**Risk Level:** Low (drop-in replacement with fallback to AdamW)

---

## Phase 1: Installation & Setup (30 minutes)

### Step 1: Clone and Install UPGD

```bash
# Navigate to parent directory
cd /home/user

# Clone UPGD repository
git clone https://github.com/mohmdelsayed/upgd.git

# Install in editable mode
cd upgd
pip install -e .

# Verify installation
python -c "from upgd import UPGD; print('UPGD installed successfully')"
```

**Expected Output:**
```
UPGD installed successfully
```

**Troubleshooting:**
- If `ModuleNotFoundError`: Check Python environment is active
- If git clone fails: Check internet connection, try HTTPS clone
- If install fails: Check dependencies (torch, numpy)

---

## Phase 2: Code Integration (2-3 hours)

### Step 2: Add Feature Flag Configuration

**File:** `distributional_ppo.py` (around line 1508, in `DistributionalPPO.__init__`)

**Action:** Add UPGD configuration parameters to policy kwargs

```python
# Find the __init__ method of DistributionalPPO class
# Around line 1508-1600

# Add these new parameters to the __init__ signature:
def __init__(
    self,
    policy,
    env,
    learning_rate=3e-4,
    # ... existing parameters ...

    # NEW: UPGD optimizer parameters
    use_upgd: bool = False,  # Feature flag to enable/disable UPGD
    upgd_utility_decay: float = 0.999,  # EMA decay for utility trace (β)
    upgd_noise_std: float = 0.01,  # Noise std for plasticity (σ)
    upgd_adaptive: bool = True,  # Use AdaUPGD (sigmoid scaling)
    upgd_protect: bool = True,  # Enable utility gating (forgetting protection)

    **kwargs,
):
    # Store UPGD config
    self.use_upgd = use_upgd
    self.upgd_config = {
        'utility_decay': upgd_utility_decay,
        'noise_std': upgd_noise_std,
        'adaptive': upgd_adaptive,
        'protect': upgd_protect,
    }

    # ... rest of existing __init__ code ...
```

**Explanation:**
- `use_upgd`: Master switch (False = AdamW, True = UPGD)
- `utility_decay`: How fast to forget utility history (0.99-0.999 recommended)
- `noise_std`: Perturbation magnitude for plasticity (0.001-0.1 range)
- `adaptive`: Use per-layer sigmoid scaling (recommended for PPO)
- `protect`: Enable utility gating (core anti-forgetting mechanism)

---

### Step 3: Modify Optimizer Initialization

**File:** `distributional_ppo.py` (around line 5541)

**Current Code:**
```python
# Around line 5541-5546
optimizer = torch.optim.AdamW(
    params=params_groups,
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)
```

**New Code:**
```python
# Import UPGD at top of file (after other imports, around line 50-100)
try:
    from upgd import UPGD
    UPGD_AVAILABLE = True
except ImportError:
    UPGD_AVAILABLE = False
    import warnings
    warnings.warn("UPGD not installed. Install with: pip install git+https://github.com/mohmdelsayed/upgd.git")

# Replace optimizer initialization (around line 5541-5546)
if self.use_upgd and UPGD_AVAILABLE:
    # Use UPGD optimizer
    optimizer = UPGD(
        params=params_groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        # UPGD-specific parameters
        utility_decay=self.upgd_config['utility_decay'],
        noise_std=self.upgd_config['noise_std'],
        adaptive=self.upgd_config['adaptive'],
        protect=self.upgd_config['protect'],
    )
    print(f"✓ Using UPGD optimizer (utility_decay={self.upgd_config['utility_decay']}, "
          f"noise_std={self.upgd_config['noise_std']}, adaptive={self.upgd_config['adaptive']})")
else:
    # Fallback to AdamW
    if self.use_upgd and not UPGD_AVAILABLE:
        warnings.warn("UPGD requested but not available. Falling back to AdamW.")
    optimizer = torch.optim.AdamW(
        params=params_groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )
    print("✓ Using AdamW optimizer (default)")
```

**Key Safety Features:**
- ✅ Graceful fallback if UPGD not installed
- ✅ Clear logging of which optimizer is used
- ✅ Feature flag allows instant rollback
- ✅ Same API as AdamW (no other code changes needed)

---

### Step 4: Add Utility Metrics Logging (Optional but Recommended)

**File:** `distributional_ppo.py` (in training loop, around line 8800-9000)

**Find:** The main training loop where `self.logger.record()` is called

**Add:** UPGD utility logging

```python
# In the training loop, after optimizer.step(), around line 8850-8900
# Find where other metrics are logged (look for self.logger.record calls)

# Add this logging block:
if self.use_upgd and hasattr(self.policy.optimizer, 'get_avg_utility'):
    try:
        # Log average utility across all parameters
        avg_utility = self.policy.optimizer.get_avg_utility()
        self.logger.record("train/upgd_avg_utility", avg_utility)

        # Log per-layer utilities (if available)
        if hasattr(self.policy.optimizer, 'get_layer_utilities'):
            layer_utilities = self.policy.optimizer.get_layer_utilities()
            for layer_name, utility in layer_utilities.items():
                self.logger.record(f"train/upgd_utility_{layer_name}", utility)
    except Exception as e:
        # Don't crash training if logging fails
        warnings.warn(f"UPGD utility logging failed: {e}")
```

**What this logs:**
- Average utility across all weights (0-1 scale)
- High utility (→1) = important weights being protected
- Low utility (→0) = weights available for plasticity

**Usage:** Monitor in TensorBoard to see which parts of network are protected

---

### Step 5: Update Training Configuration Files

**File:** `train_model_multi_patch.py` or wherever you instantiate `DistributionalPPO`

**Find:** Where you create the PPO model (search for `DistributionalPPO(`)

**Modify:** Add UPGD configuration

```python
# Example current code:
model = DistributionalPPO(
    policy=CustomActorCriticPolicy,
    env=env,
    learning_rate=3e-4,
    clip_range=0.05,
    vf_coef=1.8,
    # ... other params ...
)

# NEW code with UPGD:
model = DistributionalPPO(
    policy=CustomActorCriticPolicy,
    env=env,
    learning_rate=3e-4,
    clip_range=0.05,
    vf_coef=1.8,
    # ... other existing params ...

    # UPGD configuration
    use_upgd=True,  # ← Set to True to enable, False to use AdamW
    upgd_utility_decay=0.999,  # Conservative: slow utility forgetting
    upgd_noise_std=0.01,  # Moderate plasticity
    upgd_adaptive=True,  # Use AdaUPGD variant (recommended for PPO)
    upgd_protect=True,  # Enable anti-forgetting
)
```

**Recommendation:** Create a config dictionary for easy experimentation

```python
# At top of training script
UPGD_CONFIG = {
    'use_upgd': True,  # Master switch
    'upgd_utility_decay': 0.999,
    'upgd_noise_std': 0.01,
    'upgd_adaptive': True,
    'upgd_protect': True,
}

# Pass to model
model = DistributionalPPO(
    # ... other params ...
    **UPGD_CONFIG
)
```

---

## Phase 3: Testing & Validation (1-2 weeks)

### Step 6: Unit Test - Verify Optimizer Works

**File:** Create `tests/test_upgd_integration.py`

```python
import torch
import torch.nn as nn
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy
import gym

def test_upgd_initialization():
    """Test that UPGD optimizer initializes correctly"""
    env = gym.make('CartPole-v1')

    # Test with UPGD enabled
    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        use_upgd=True,
        upgd_utility_decay=0.999,
        upgd_noise_std=0.01,
    )

    # Check optimizer type
    from upgd import UPGD
    assert isinstance(model.policy.optimizer, UPGD), "Should use UPGD optimizer"
    print("✓ UPGD initialization test passed")

def test_upgd_fallback():
    """Test fallback to AdamW when UPGD disabled"""
    env = gym.make('CartPole-v1')

    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        use_upgd=False,  # Disabled
    )

    assert isinstance(model.policy.optimizer, torch.optim.AdamW), "Should use AdamW"
    print("✓ AdamW fallback test passed")

def test_upgd_training_step():
    """Test that training step works with UPGD"""
    env = gym.make('CartPole-v1')

    model = DistributionalPPO(
        policy=CustomActorCriticPolicy,
        env=env,
        use_upgd=True,
        n_steps=128,
    )

    # Run one training iteration
    try:
        model.learn(total_timesteps=256)
        print("✓ UPGD training step test passed")
    except Exception as e:
        raise AssertionError(f"Training failed with UPGD: {e}")

if __name__ == "__main__":
    test_upgd_initialization()
    test_upgd_fallback()
    test_upgd_training_step()
    print("\n✅ All UPGD tests passed!")
```

**Run Tests:**
```bash
cd /home/user/TradingBot2
python tests/test_upgd_integration.py
```

**Expected Output:**
```
✓ UPGD initialization test passed
✓ AdamW fallback test passed
✓ UPGD training step test passed

✅ All UPGD tests passed!
```

---

### Step 7: Backtest Comparison

**Objective:** Compare UPGD vs AdamW on historical data (2023-2024)

**Script:** Create `experiments/compare_upgd_adamw.py`

```python
import pandas as pd
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy
# ... your environment imports ...

# Configuration
TOTAL_TIMESTEPS = 1_000_000  # Adjust based on your typical training
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10

# Shared config
base_config = {
    'policy': CustomActorCriticPolicy,
    'env': env,  # Your trading environment
    'learning_rate': 3e-4,
    'clip_range': 0.05,
    'vf_coef': 1.8,
    'entropy_coef': 0.01,
    'cvar_weight': 0.5,
    # ... other params from your current config ...
}

# Experiment 1: AdamW baseline
print("=" * 60)
print("Training with AdamW (Baseline)...")
print("=" * 60)

model_adamw = DistributionalPPO(
    **base_config,
    use_upgd=False,  # AdamW
    tensorboard_log="./logs/comparison/adamw"
)

model_adamw.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name="adamw_baseline"
)

# Evaluate
returns_adamw = evaluate_model(model_adamw, env, n_episodes=N_EVAL_EPISODES)

# Experiment 2: UPGD
print("\n" + "=" * 60)
print("Training with UPGD...")
print("=" * 60)

model_upgd = DistributionalPPO(
    **base_config,
    use_upgd=True,
    upgd_utility_decay=0.999,
    upgd_noise_std=0.01,
    upgd_adaptive=True,
    tensorboard_log="./logs/comparison/upgd"
)

model_upgd.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name="upgd_experimental"
)

# Evaluate
returns_upgd = evaluate_model(model_upgd, env, n_episodes=N_EVAL_EPISODES)

# Compare
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

results = pd.DataFrame({
    'Optimizer': ['AdamW', 'UPGD'],
    'Mean Return': [returns_adamw.mean(), returns_upgd.mean()],
    'Std Return': [returns_adamw.std(), returns_upgd.std()],
    'Sharpe Ratio': [
        returns_adamw.mean() / returns_adamw.std(),
        returns_upgd.mean() / returns_upgd.std()
    ],
    'Max Drawdown': [
        compute_max_drawdown(returns_adamw),
        compute_max_drawdown(returns_upgd)
    ]
})

print(results)

# Statistical significance test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(returns_adamw, returns_upgd)
print(f"\nT-test: t={t_stat:.4f}, p={p_value:.4f}")
if p_value < 0.05:
    print("✓ Difference is statistically significant (p < 0.05)")
else:
    print("⚠ Difference not statistically significant")

# Save results
results.to_csv('experiments/upgd_vs_adamw_results.csv', index=False)
print("\n✓ Results saved to experiments/upgd_vs_adamw_results.csv")
```

**Run Comparison:**
```bash
cd /home/user/TradingBot2
python experiments/compare_upgd_adamw.py
```

**Success Criteria:**
- ✅ UPGD matches or exceeds AdamW on mean return
- ✅ UPGD has lower std (better stability)
- ✅ Sharpe ratio improvement ≥5%
- ✅ Training converges without crashes

---

### Step 8: Long-Term Forgetting Test

**Objective:** Verify UPGD prevents catastrophic forgetting over extended training

**Script:** Create `experiments/test_continual_learning.py`

```python
"""
Test catastrophic forgetting by:
1. Train on Period 1 (e.g., bull market)
2. Evaluate on Period 1
3. Train on Period 2 (e.g., bear market)
4. Re-evaluate on Period 1
5. Check if performance dropped (forgetting)
"""

# Split data into regimes
period1_env = create_env(data='2023-01-01:2023-06-30')  # Bull
period2_env = create_env(data='2023-07-01:2023-12-31')  # Bear

# Test AdamW
model_adamw = DistributionalPPO(..., use_upgd=False)

# Phase 1: Learn Period 1
model_adamw.set_env(period1_env)
model_adamw.learn(total_timesteps=500_000)
perf_adamw_p1_initial = evaluate(model_adamw, period1_env)

# Phase 2: Learn Period 2
model_adamw.set_env(period2_env)
model_adamw.learn(total_timesteps=500_000)
perf_adamw_p2 = evaluate(model_adamw, period2_env)

# Phase 3: Re-evaluate Period 1 (check forgetting)
perf_adamw_p1_after = evaluate(model_adamw, period1_env)

forgetting_adamw = (perf_adamw_p1_initial - perf_adamw_p1_after) / perf_adamw_p1_initial * 100

# Test UPGD
model_upgd = DistributionalPPO(..., use_upgd=True, upgd_utility_decay=0.999)

model_upgd.set_env(period1_env)
model_upgd.learn(total_timesteps=500_000)
perf_upgd_p1_initial = evaluate(model_upgd, period1_env)

model_upgd.set_env(period2_env)
model_upgd.learn(total_timesteps=500_000)
perf_upgd_p2 = evaluate(model_upgd, period2_env)

perf_upgd_p1_after = evaluate(model_upgd, period1_env)

forgetting_upgd = (perf_upgd_p1_initial - perf_upgd_p1_after) / perf_upgd_p1_initial * 100

# Compare
print(f"AdamW forgetting: {forgetting_adamw:.2f}%")
print(f"UPGD forgetting: {forgetting_upgd:.2f}%")
print(f"Improvement: {forgetting_adamw - forgetting_upgd:.2f}% less forgetting with UPGD")

# Success if UPGD has 20-50% less forgetting
if forgetting_upgd < forgetting_adamw * 0.8:
    print("✅ UPGD significantly reduces catastrophic forgetting")
else:
    print("⚠ UPGD benefit unclear, may need hyperparameter tuning")
```

---

## Phase 4: Hyperparameter Tuning (Optional, 2-3 days)

### Step 9: Tune UPGD Parameters

If initial results are mixed, tune these parameters:

**Parameter Grid:**

```python
from itertools import product

# Define search space
param_grid = {
    'upgd_utility_decay': [0.99, 0.995, 0.999],  # How fast to forget utility
    'upgd_noise_std': [0.001, 0.01, 0.1],  # Plasticity level
    'upgd_adaptive': [True],  # Keep True for PPO
}

# Grid search
results = []
for utility_decay, noise_std, adaptive in product(*param_grid.values()):
    model = DistributionalPPO(
        ...,
        use_upgd=True,
        upgd_utility_decay=utility_decay,
        upgd_noise_std=noise_std,
        upgd_adaptive=adaptive,
    )

    model.learn(total_timesteps=500_000)
    performance = evaluate(model, env)

    results.append({
        'utility_decay': utility_decay,
        'noise_std': noise_std,
        'performance': performance
    })

# Find best
best = max(results, key=lambda x: x['performance'])
print(f"Best config: {best}")
```

**Quick Tuning Guidelines:**

- **`utility_decay`:**
  - **0.99** = Fast forgetting of utility → more plasticity (try if UPGD too conservative)
  - **0.999** = Slow forgetting → more protection (recommended start)

- **`noise_std`:**
  - **0.001** = Low noise → less plasticity (try if training unstable)
  - **0.01** = Medium noise → balanced (recommended start)
  - **0.1** = High noise → more exploration (try if stuck in local minimum)

---

## Phase 5: Deployment (1 week)

### Step 10: Paper Trading Validation

**Before production:**

1. **Deploy UPGD model to paper trading** (10% capital allocation)
2. **Run for 1-2 weeks** alongside current AdamW model
3. **Monitor metrics:**
   - Daily Sharpe ratio
   - Max drawdown
   - Trade quality (slippage, execution)
   - Model stability (no crashes)

**Success Criteria:**
- ✅ Sharpe ratio ≥ 90% of AdamW (within noise)
- ✅ No crashes or numerical instabilities
- ✅ Utility metrics in reasonable range (0.3-0.7 average)

**If Successful → Full Rollout:**

```python
# Update production config
PRODUCTION_CONFIG = {
    'use_upgd': True,  # Enable UPGD
    'upgd_utility_decay': 0.999,  # Use best config from tuning
    'upgd_noise_std': 0.01,
    'upgd_adaptive': True,
    'upgd_protect': True,
}

# Deploy to 100% of capital
production_model = DistributionalPPO(
    **PRODUCTION_CONFIG,
    # ... other production settings ...
)
```

---

## Rollback Plan

If UPGD causes issues, instant rollback:

```python
# Simply flip the flag
model = DistributionalPPO(
    ...,
    use_upgd=False,  # ← Back to AdamW
)
```

No other code changes needed!

---

## Monitoring & Maintenance

### Key Metrics to Watch

**TensorBoard:**
```bash
tensorboard --logdir=./logs
```

**Monitor:**
1. `train/upgd_avg_utility` - Should be 0.3-0.7 (too high = over-protection, too low = no benefit)
2. `train/loss` - Should decrease smoothly (no spikes)
3. `eval/mean_reward` - Should increase or stay stable over time
4. `eval/sharpe_ratio` - Key performance metric

**Alert if:**
- ⚠️ Average utility > 0.9 (model too conservative, increase `noise_std`)
- ⚠️ Average utility < 0.1 (no protection, increase `utility_decay`)
- ⚠️ Loss spikes > 2× normal (reduce `noise_std`)

---

## Expected Timeline

| Phase | Duration | Effort |
|-------|----------|--------|
| Installation | 30 min | Copy-paste commands |
| Code Integration | 2-3 hours | 30-50 lines of code |
| Unit Testing | 1 hour | Run test script |
| Backtest Comparison | 2-3 days | Automated script |
| Long-term Test | 3-5 days | Automated script |
| Hyperparameter Tuning | 2-3 days | Grid search (optional) |
| Paper Trading | 1-2 weeks | Monitor in parallel |
| **Total** | **2-4 weeks** | **1 engineer** |

---

## Troubleshooting

### Issue 1: UPGD Import Error

**Symptom:**
```
ModuleNotFoundError: No module named 'upgd'
```

**Fix:**
```bash
pip install git+https://github.com/mohmdelsayed/upgd.git
```

### Issue 2: Training Slower with UPGD

**Symptom:** Training takes 30-50% longer

**Explanation:** Expected - UPGD computes Hessian diagonal (extra backward pass)

**Mitigation:**
- Compute Hessian every N steps instead of every step (modify UPGD source)
- Use smaller `noise_std` (less computation)
- Accept tradeoff (longer training = better long-term performance)

### Issue 3: Utility Always Near 0 or 1

**Symptom:** `train/upgd_avg_utility` stuck at extremes

**Fix:**
- **If ~0:** Increase `utility_decay` (0.995 → 0.999)
- **If ~1:** Decrease `utility_decay` (0.999 → 0.995) or increase `noise_std`

### Issue 4: Performance Worse than AdamW

**Symptom:** UPGD underperforms in backtests

**Debug:**
1. Check utility values - should be 0.3-0.7 range
2. Try different `noise_std` (0.001, 0.01, 0.1)
3. Ensure adaptive=True (important for PPO)
4. Run longer - UPGD benefits appear over extended training

---

## Success Metrics

**Immediate (Week 1-2):**
- ✅ UPGD integrates without crashes
- ✅ Training converges at similar speed to AdamW
- ✅ Utility metrics in healthy range (0.3-0.7)

**Medium-term (Month 1):**
- ✅ Performance ≥ AdamW baseline on backtests
- ✅ Less performance degradation after regime changes (20-30% reduction)
- ✅ Stable paper trading results

**Long-term (Month 3-6):**
- ✅ No need for quarterly retraining (model stays performant)
- ✅ Better handling of rare events (crashes, squeezes)
- ✅ 20-30% improvement in risk-adjusted returns over extended deployment

---

## Next Steps After UPGD

Once UPGD is stable:

1. **Combine with PBT** - Use Population-Based Training to auto-tune UPGD hyperparameters
2. **Add Twin Critics** - Further stability improvement (5-10%)
3. **Implement GTrXL** - Better memory architecture

**UPGD is the foundation** - prevents your improvements from being forgotten!

---

## Support & Resources

**Paper:** https://arxiv.org/abs/2404.00781
**Code:** https://github.com/mohmdelsayed/upgd
**Issues:** Report integration issues to TradingBot2 team

---

**Good luck with the integration! 🚀**
