# Training Pipeline: Quick File Reference

## Core Training Files

### 1. Entry Points & Orchestration

| File | Lines | Purpose |
|------|-------|---------|
| **service_train.py** | 144-267 | Main training orchestration, NaN filtering, dataset export |
| **train_model_multi_patch.py** | 1-200 | HPO (Optuna), environment setup, training loop initialization |
| **distributional_ppo.py** | 1459+ | Core RL algorithm (DistributionalPPO class), training loop (line 6508) |

### 2. Data Preparation & Leakage Prevention

| File | Lines | Purpose |
|------|-------|---------|
| **build_training_table.py** | 22-77 | Asof-join merging, leakguard integration, label construction |
| **leakguard.py** | 1-232 | Forward-looking bias prevention, decision_delay_ms, ffill validation |
| **labels.py** | 34-102 | Target variable builder, horizon returns, merge_asof logic |

### 3. Feature Engineering & Preprocessing

| File | Lines | Purpose |
|------|-------|---------|
| **transformers.py** | 704+ | OnlineFeatureTransformer, feature computation (MA, RSI, GARCH, etc) |
| **feature_pipe.py** | 1-200 | Feature pipeline orchestration, warmup/fit patterns |
| **features_pipeline.py** | 1-100 | Feature config and specs |
| **obs_builder.pyx** | 1-150 | Observation construction, tanh normalization, data validation |

### 4. Normalization & Scaling

| File | Lines | Purpose |
|------|-------|---------|
| **normalization_utils.py** | 1-323 | Analysis tools, saturation check, KS-test validation |
| **obs_builder.pyx** | 1-70 | Tanh normalization implementation, NaN handling |

### 5. Loss Computation & Training Loop

| File | Lines | Purpose |
|------|-------|---------|
| **distributional_ppo.py** | 2435-2484 | _quantile_huber_loss (critic loss) |
| **distributional_ppo.py** | 6508+ | train() method (full training loop) |
| **distributional_ppo.py** | 7862-7908 | Policy loss computation (PPO + BC + KL) |
| **distributional_ppo.py** | 8387-8751 | Critic loss computation (quantile or categorical) |
| **distributional_ppo.py** | 8753-8760 | CVaR penalty computation |
| **distributional_ppo.py** | 8761-8772 | Total loss composition & backpropagation |
| **distributional_ppo.py** | 8809-8844 | Gradient clipping & optimizer step |

### 6. Validation & Evaluation

| File | Lines | Purpose |
|------|-------|---------|
| **service_eval.py** | 1-100 | Evaluation service |
| **evaluate_policy_custom_cython.py** | 1-100 | Fast Cython-based policy evaluation |
| **services/metrics.py** | 1-100 | Metrics computation |
| **data_validation.py** | 1-100 | Data integrity checks |

---

## Critical Data Flow

```
Raw OHLCV Data
    ↓
build_training_table.py: Asof-join sources + labels
    ↓
leakguard.py: Add decision_ts, validate ffill gaps
    ↓
labels.py: Build horizon returns
    ↓
service_train.py: Load, warmup, fit pipeline
    ↓
transformers.py: Compute features (MA, RSI, GARCH, etc)
    ↓
feature_pipe.py: Transform features
    ↓
obs_builder.pyx: Normalize (tanh) & build observations
    ↓
Data Validation:
  - Check NaN targets: service_train.py:214-226
  - Check index alignment: service_train.py:181-212
  - Log feature fill: service_train.py:81-142
    ↓
distributional_ppo.py: train()
  - Load rollout buffer
  - Compute policy loss: line 7862
  - Compute critic loss: line 8387
  - Compute CVaR loss: line 8753
  - Total loss: line 8761
  - Backward: line 8772
  - Clip gradients: line 8809
  - Optimizer step: line 8844
```

---

## Key Parameters to Monitor

### Leakage Prevention (leakguard.py)
- **decision_delay_ms** (default: 8000ms) → Must be >= 8000ms
- **min_lookback_ms** → Ensures sufficient history

### Feature Normalization (obs_builder.pyx)
- **Normalization**: tanh(raw_value) → [-1, 1]
- **Saturation check**: % values at |x| > 0.95

### Loss Components (distributional_ppo.py)
- **policy_loss** → PPO clipping loss
- **critic_loss** → Quantile Huber or categorical cross-entropy
- **cvar_term** → Risk penalty
- **entropy_loss** → Exploration bonus
- **Total loss** = policy + ent_coef*entropy + vf_coef*critic + cvar_term

### Training Metrics (distributional_ppo.py)
- **explained_variance** → Line 3129
- **grad_norm** → Line 8817
- **approx_kl** → Line 8872
- **returns statistics** → Line 6637

---

## Critical Validation Points

### 1. NaN Filtering (MUST DO)
```
Location: service_train.py:214-226
Check: valid_mask = y.notna()
Action: Remove rows where y is NaN
```

### 2. Index Alignment (MUST DO)
```
Location: service_train.py:181-212
Check: len(X) == len(y) after alignment
Action: Reset indices to [0, 1, 2, ...]
```

### 3. Forward-Looking Bias (MUST VERIFY)
```
Location: leakguard.py:58
Check: decision_delay_ms >= 8000
Action: Warn if below recommendation
```

### 4. Feature Distribution (SHOULD CHECK)
```
Location: normalization_utils.py:200
Function: validate_normalization_consistency(train_obs, val_obs)
Method: KS-test (p-value < 0.01 = shift detected)
```

### 5. Loss Computation (MONITOR)
```
Location: distributional_ppo.py:8761-8772
Check: loss must be finite (not NaN/Inf)
Check: gradients must be finite after backward()
```

---

## File Line Numbers Summary

| File | Key Lines | Purpose |
|------|-----------|---------|
| service_train.py | 144 | run() entry point |
| service_train.py | 81-142 | Feature statistics logging |
| service_train.py | 177-212 | Index alignment |
| service_train.py | 214-239 | NaN target filtering |
| leakguard.py | 58 | RECOMMENDED_MIN_DELAY_MS constant |
| leakguard.py | 60-93 | Delay validation logic |
| transformers.py | 704 | OnlineFeatureTransformer class |
| transformers.py | 1113 | apply_offline_features function |
| distributional_ppo.py | 1459 | DistributionalPPO class |
| distributional_ppo.py | 2435 | _quantile_huber_loss method |
| distributional_ppo.py | 6508 | train() method |
| distributional_ppo.py | 7862 | Policy loss computation |
| distributional_ppo.py | 8387 | Critic loss computation |
| distributional_ppo.py | 8753 | CVaR loss computation |
| distributional_ppo.py | 8761 | Total loss assembly |
| distributional_ppo.py | 8772 | loss.backward() call |
| distributional_ppo.py | 8809 | Gradient clipping |
| distributional_ppo.py | 8844 | Optimizer step |
| normalization_utils.py | 200 | validate_normalization_consistency |
| obs_builder.pyx | 1-70 | Normalization & validation functions |

