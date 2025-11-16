# Migration Guide: Observation Vector 56 → 62 Features

**Date**: 2025-11-16
**Breaking Change**: YES
**Impact**: HIGH - All trained models incompatible

---

## Summary

This migration adds **validity flags** for ALL technical indicators, increasing observation vector size from **56 to 62 features** (+6 flags).

### What Changed:

| Feature | Before (56) | After (62) | Change |
|---------|-------------|------------|--------|
| ma5 + is_ma5_valid | ✅ Had flag | ✅ Still has flag | No change |
| ma20 + is_ma20_valid | ✅ Had flag | ✅ Still has flag | No change |
| **rsi14** | ❌ No flag | ✅ **+is_rsi_valid** | **NEW** |
| **macd** | ❌ No flag | ✅ **+is_macd_valid** | **NEW** |
| **macd_signal** | ❌ No flag | ✅ **+is_macd_signal_valid** | **NEW** |
| **momentum** | ❌ No flag | ✅ **+is_momentum_valid** | **NEW** |
| **cci** | ❌ No flag | ✅ **+is_cci_valid** | **NEW** |
| **obv** | ❌ No flag | ✅ **+is_obv_valid** | **NEW** |

---

## Why This Change?

### Problem: Ambiguous Fallback Values

**Before** (without validity flags):
```python
# RSI example:
rsi = 50.0  # Could mean:
            # 1. No data yet (first 14 bars) → fallback
            # 2. Neutral zone → real RSI value
# Model CANNOT distinguish these cases!

# MACD example:
macd = 0.0  # Could mean:
            # 1. No data yet (first 26 bars) → fallback
            # 2. No divergence (EMA12 = EMA26) → real value
# Model CANNOT distinguish these cases!
```

**After** (with validity flags):
```python
# RSI example:
rsi = 50.0, is_rsi_valid = 0.0  → No data (warmup period)
rsi = 50.0, is_rsi_valid = 1.0  → Real neutral RSI

# MACD example:
macd = 0.0, is_macd_valid = 0.0  → No data (warmup period)
macd = 0.0, is_macd_valid = 1.0  → Real no-divergence
```

### Solution Benefits:

1. ✅ **Model knows which data to trust** (can ignore invalid indicators)
2. ✅ **Production-ready** (handles MarketSimulator=None gracefully)
3. ✅ **Full consistency** (all indicators follow same pattern)
4. ✅ **Analyzable** (can measure indicator importance)

---

## New Feature Indices

### Feature Mapping Changes:

| Index | Feature (Before 56) | Index | Feature (After 62) | Shift |
|-------|---------------------|-------|-------------------|-------|
| 0-6 | price, volumes, ma5, is_ma5_valid, ma20, is_ma20_valid | 0-6 | Same | +0 |
| 7 | rsi14 | 7 | rsi14 | +0 |
| **-** | **-** | **8** | **is_rsi_valid** | **NEW** |
| 8 | macd | 9 | macd | +1 |
| **-** | **-** | **10** | **is_macd_valid** | **NEW** |
| 9 | macd_signal | 11 | macd_signal | +2 |
| **-** | **-** | **12** | **is_macd_signal_valid** | **NEW** |
| 10 | momentum | 13 | momentum | +3 |
| **-** | **-** | **14** | **is_momentum_valid** | **NEW** |
| 11 | atr | 15 | atr | +4 |
| 12 | cci | 16 | cci | +4 |
| **-** | **-** | **17** | **is_cci_valid** | **NEW** |
| 13 | obv | 18 | obv | +5 |
| **-** | **-** | **19** | **is_obv_valid** | **NEW** |
| 14-55 | Derived features, state, etc. | 20-61 | Same features | +6 |

**Summary**: All features after `rsi14` (index 7) are shifted by +1 to +6 positions.

---

## Migration Steps

### 1. Code Changes

#### ✅ Already Updated Files:
- `obs_builder.pyx` - Added 4 validity flags (rsi, momentum, cci, obv)
- `tests/test_technical_indicators_in_obs.py` - Updated to 62
- `tests/test_volume_validation.py` - Updated to 62
- `tests/test_price_validation.py` - Updated to 62
- `tests/test_bollinger_bands_validation.py` - Updated to 62
- `tests/test_mediator_integration.py` - Updated to 62
- `tests/test_prev_price_ret_bar.py` - Updated to 62
- `OBSERVATION_MAPPING.md` - Updated to 62
- `FEATURE_MAPPING_62.md` - Updated (renamed from 56)
- `AUDIT_REPORT.md` - Updated to 62

#### ⚠️ Requires Manual Update:
- Any custom scripts that hardcode observation size
- Training scripts that specify `observation_space=(56,)`
- Checkpoint loading code (if it validates dimensions)

### 2. Rebuild Cython Modules

**CRITICAL**: Must rebuild after obs_builder.pyx changes:

```bash
python setup.py build_ext --inplace
```

### 3. Retrain Models

**CRITICAL**: All existing checkpoints are incompatible!

```bash
# Old models expect 56 inputs, will fail with:
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x62 and 56xN)

# Solution: Retrain from scratch
python train_ppo_distributed.py \
  --config configs/config_train.yaml \
  --learning-rate 5e-5 \
  --batch-size 256
```

### 4. Verify Tests Pass

```bash
pytest tests/test_technical_indicators_in_obs.py -v
pytest tests/test_volume_validation.py -v
pytest tests/test_price_validation.py -v
```

---

## Compatibility Matrix

| Component | Version Before | Version After | Compatible? |
|-----------|---------------|---------------|-------------|
| **obs_builder.pyx** | 56 features | 62 features | ❌ BREAKING |
| **Trained models** | Input(56) | Input(62) | ❌ INCOMPATIBLE |
| **Test suite** | Expects 56 | Expects 62 | ✅ Updated |
| **Documentation** | Says 56 | Says 62 | ✅ Updated |
| **Data pipeline** | N/A | N/A | ✅ No change |

---

## Rollback Plan

If you need to rollback:

```bash
git revert <commit-hash>
python setup.py build_ext --inplace
# Use old model checkpoints (input_size=56)
```

---

## Testing Checklist

- [ ] Rebuild Cython modules (`setup.py build_ext --inplace`)
- [ ] Run unit tests (`pytest tests/ -v`)
- [ ] Train for 1 episode to verify observation shape
- [ ] Check PPO model accepts observation (62,)
- [ ] Verify no NaN/Inf in observations during training
- [ ] Monitor first 50 bars (warmup period) - validity flags should be 0.0

---

## Example: Observation During Warmup

### Bar 0 (First bar):
```python
obs[7] = 50.0   # rsi14 (fallback)
obs[8] = 0.0    # is_rsi_valid ← INVALID
obs[9] = 0.0    # macd (fallback)
obs[10] = 0.0   # is_macd_valid ← INVALID
obs[11] = 0.0   # macd_signal (fallback)
obs[12] = 0.0   # is_macd_signal_valid ← INVALID
obs[13] = 0.0   # momentum (fallback)
obs[14] = 0.0   # is_momentum_valid ← INVALID
obs[16] = 0.0   # cci (fallback)
obs[17] = 0.0   # is_cci_valid ← INVALID
obs[18] = 0.0   # obv (fallback)
obs[19] = 0.0   # is_obv_valid ← INVALID
```

### Bar 30 (All indicators valid):
```python
obs[7] = 48.2   # rsi14 (real value)
obs[8] = 1.0    # is_rsi_valid ← VALID
obs[9] = -5.3   # macd (real value)
obs[10] = 1.0   # is_macd_valid ← VALID
obs[11] = -3.1  # macd_signal (real value)
obs[12] = 1.0   # is_macd_signal_valid ← VALID
obs[13] = 120.5 # momentum (real value)
obs[14] = 1.0   # is_momentum_valid ← VALID
obs[16] = 45.0  # cci (real value)
obs[17] = 1.0   # is_cci_valid ← VALID
obs[18] = 1e6   # obv (real value)
obs[19] = 1.0   # is_obv_valid ← VALID
```

---

## FAQ

### Q: Do I need to retrain all models?
**A**: YES. Old checkpoints expect 56 inputs, new system provides 62.

### Q: Can I use old data?
**A**: YES. Data format unchanged, only observation vector structure changed.

### Q: Will performance improve?
**A**: Likely YES. Model can now distinguish valid data from warmup period, leading to better decisions in early episodes.

### Q: What if I don't have time to retrain?
**A**: Use `git revert` to rollback to 56-feature version (not recommended long-term).

### Q: How long to retrain?
**A**: Same as before - depends on your hardware and training duration (e.g., 100k steps).

---

## References

- **Commit**: See git log for exact commit hash
- **Original Analysis**: `MACD_VALIDATION_AUDIT.md`
- **Critical Review**: `CRITICAL_REVIEW_MACD_FIX.md`
- **Architecture Docs**: `OBSERVATION_MAPPING.md`, `FEATURE_MAPPING_62.md`

---

## Support

If you encounter issues:

1. Check Cython rebuild completed successfully
2. Verify tests pass
3. Check observation shape in training logs
4. Review error messages for dimension mismatches

---

**Migration completed**: 2025-11-16
**Author**: Claude
**Status**: Ready for testing
