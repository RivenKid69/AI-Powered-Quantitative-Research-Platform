# Twin Critics Integration

## Overview

Twin Critics is a technique borrowed from TD3 and SAC algorithms that uses two independent value networks to reduce overestimation bias in value function estimation. This implementation integrates Twin Critics into the DistributionalPPO algorithm.

## Background

### Problem: Overestimation Bias
Standard reinforcement learning algorithms often overestimate action values, especially in stochastic environments like trading. This occurs because the same network is used for both:
1. Selecting actions (policy)
2. Evaluating those actions (critic)

### Solution: Twin Critics
By maintaining two independent critic networks and using the minimum of their estimates, we get a more conservative (less biased) value estimate. This is the core idea behind TD3 (Twin Delayed DDPG) and is also used in SAC (Soft Actor-Critic).

### Research Support
- **PDPPO (2025)**: Post-Decision PPO with Dual Critics showed ~2x better performance in stochastic environments
- **DNA (2022)**: Dual Network Architecture for PPO demonstrated reduced negative interference between policy and value learning
- **TD3 (2018)**: Addressing Function Approximation Error in Actor-Critic Methods

## Architecture

### Without Twin Critics (Default)
```
[Observation] → [Features] → [LSTM] → [MLP] → [Critic Head] → [Value]
```

### With Twin Critics (Enabled)
```
[Observation] → [Features] → [LSTM] → [MLP] → [Critic Head 1] → [Value 1]
                                              ↘ [Critic Head 2] → [Value 2]

Target Value = min(Value 1, Value 2)
```

## Configuration

### Enable Twin Critics

Add `use_twin_critics: True` to your architecture parameters:

```python
arch_params = {
    'hidden_dim': 64,
    'critic': {
        'distributional': True,  # or False for categorical
        'num_quantiles': 32,     # if distributional=True
        'huber_kappa': 1.0,
        'use_twin_critics': True,  # ← Enable Twin Critics
    }
}
```

### YAML Configuration

```yaml
model:
  arch_params:
    hidden_dim: 64
    critic:
      distributional: true
      num_quantiles: 32
      huber_kappa: 1.0
      use_twin_critics: true  # ← Enable Twin Critics
```

## Features

### Supported Modes

Twin Critics works with both:
1. **Quantile Critic** (distributional=True): Two independent quantile value heads
2. **Categorical Critic** (distributional=False): Two independent categorical value heads

### Key Benefits

1. **Reduced Overestimation**: Pessimistic value estimates reduce harmful overoptimism
2. **Better Generalization**: More robust value estimates in stochastic environments
3. **Improved Stability**: Dual critics provide redundancy and stability
4. **Trading-Specific**: Particularly beneficial for volatile trading environments

### Implementation Details

- Both critics share the same input features (LSTM/MLP backbone)
- Each critic has independent parameters (separate linear heads)
- Both critics are trained with the same targets
- Minimum of both estimates is used for advantage calculation
- Fully backward compatible (disabled by default)

## Usage Examples

### Basic Training with Twin Critics

```python
from distributional_ppo import DistributionalPPO
from custom_policy_patch1 import CustomActorCriticPolicy

arch_params = {
    'hidden_dim': 64,
    'lstm_hidden_size': 64,
    'critic': {
        'distributional': True,
        'num_quantiles': 32,
        'use_twin_critics': True,  # Enable Twin Critics
    }
}

model = DistributionalPPO(
    CustomActorCriticPolicy,
    env,
    arch_params=arch_params,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    learning_rate=0.0003,
    verbose=1,
)

model.learn(total_timesteps=1_000_000)
```

### With Variance Gradient Scaling

Twin Critics is fully compatible with other features:

```python
arch_params = {
    'hidden_dim': 64,
    'critic': {
        'distributional': True,
        'num_quantiles': 32,
        'use_twin_critics': True,
    }
}

vgs_config = {
    'enabled': True,
    'beta': 0.99,
    'alpha': 0.1,
}

model = DistributionalPPO(
    CustomActorCriticPolicy,
    env,
    arch_params=arch_params,
    vgs_config=vgs_config,
    learning_rate=0.0003,
)
```

## Testing

### Run Unit Tests

```bash
pytest tests/test_twin_critics.py -v
```

### Run Integration Tests

```bash
pytest tests/test_twin_critics_integration.py -v
```

### Test Coverage

```bash
pytest tests/test_twin_critics*.py --cov=custom_policy_patch1 --cov=distributional_ppo --cov-report=html
```

## Performance Considerations

### Computational Cost

- **Memory**: ~2x critic parameters (e.g., 32 → 64 KB for typical head)
- **Compute**: ~2x forward passes for critic during training
- **Training Time**: Minimal impact (<5% overhead in practice)

### When to Use

**Recommended for:**
- Stochastic trading environments
- High-variance reward signals
- Long training runs
- Production deployments

**Not necessary for:**
- Deterministic environments
- Quick prototyping
- Environments with low noise

## API Reference

### CustomActorCriticPolicy

**New Attributes:**
- `_use_twin_critics`: bool - Whether twin critics are enabled
- `quantile_head_2`: QuantileValueHead - Second quantile critic (if quantile mode)
- `dist_head_2`: nn.Linear - Second categorical critic (if categorical mode)
- `_value_head_module_2`: nn.Module - Second critic module reference

**New Methods:**
- `_get_value_logits_2(latent_vf)`: Get second critic's raw outputs
- `_get_twin_value_logits(latent_vf)`: Get both critics' outputs
- `_get_min_twin_values(latent_vf)`: Get minimum of both critic estimates

### DistributionalPPO

**New Methods:**
- `_twin_critics_loss(latent_vf, targets, reduction)`: Compute loss for both critics

## Backward Compatibility

Twin Critics is **fully backward compatible**:

1. **Default Behavior**: Disabled by default (`use_twin_critics=False`)
2. **No Breaking Changes**: Existing configurations work unchanged
3. **Optional Feature**: Only active when explicitly enabled
4. **Save/Load**: Models save/load correctly regardless of twin critics state

## Troubleshooting

### Common Issues

**Q: Second critic not updating**
A: Check that `_value_head_module_2` is added to optimizer in `_setup_custom_optimizer()`

**Q: NaN values in training**
A: Both critics should use same hyperparameters (learning rate, clipping, etc.)

**Q: Higher memory usage**
A: Expected - twin critics doubles critic parameters. Monitor GPU memory.

**Q: Tests failing**
A: Ensure pytest and all dependencies installed: `pip install -r requirements.txt`

## References

1. Fujimoto et al. (2018): "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)
2. Haarnoja et al. (2018): "Soft Actor-Critic Algorithms and Applications" (SAC)
3. Huang et al. (2025): "Post-Decision Proximal Policy Optimization with Dual Critic Networks" (PDPPO)
4. Ota et al. (2022): "DNA: Proximal Policy Optimization with a Dual Network Architecture"

## Contributing

When modifying Twin Critics implementation:

1. Run all tests: `pytest tests/test_twin_critics*.py -v`
2. Check code coverage: Maintain >90% coverage for new code
3. Update documentation: Keep this file in sync with implementation
4. Verify backward compatibility: Test with `use_twin_critics=False`

## License

Same as parent project (TradingBot2)
