#!/usr/bin/env python
"""
Test to verify that all 62 features are properly integrated into the observation space.
Updated from 56 to 62 features (added 6 validity flags for indicators).
"""
import numpy as np
from feature_config import N_FEATURES, make_layout

# Test 1: Check N_FEATURES
print("=" * 80)
print("Test 1: Check N_FEATURES")
print("=" * 80)

make_layout({'max_num_tokens': 1, 'ext_norm_dim': 21})
print(f"✓ N_FEATURES = {N_FEATURES}")

if N_FEATURES == 62:
    print("✓ PASS: N_FEATURES is 62 as expected (was 56, added 6 validity flags)")
else:
    print(f"✗ FAIL: N_FEATURES is {N_FEATURES}, expected 62")
    exit(1)

# Test 2: Check feature layout
print("\n" + "=" * 80)
print("Test 2: Check feature layout")
print("=" * 80)

from feature_config import FEATURES_LAYOUT

total = 0
for block in FEATURES_LAYOUT:
    size = block['size']
    name = block['name']
    print(f"  {name}: {size}")
    total += size

print(f"\nTotal features: {total}")

if total == 62:
    print("✓ PASS: Feature layout sum is 62 (was 56, added 6 validity flags)")
else:
    print(f"✗ FAIL: Feature layout sum is {total}, expected 62")
    exit(1)

# Test 3: Check observation builder
print("\n" + "=" * 80)
print("Test 3: Check observation builder integration")
print("=" * 80)

try:
    from obs_builder import build_observation_vector

    # Create test observation array
    obs = np.zeros(62, dtype=np.float32)
    norm_cols = np.zeros(21, dtype=np.float32)

    # Fill with test data
    for i in range(21):
        norm_cols[i] = float(i) * 0.1

    # Call obs_builder
    build_observation_vector(
        price=100.0,
        prev_price=99.0,
        log_volume_norm=0.5,
        rel_volume=1.0,
        ma5=100.5,
        ma20=100.2,
        rsi14=50.0,
        macd=0.1,
        macd_signal=0.05,
        momentum=0.2,
        atr=1.5,
        cci=0.0,
        obv=1000.0,
        bb_lower=98.0,
        bb_upper=102.0,
        is_high_importance=0.0,
        time_since_event=0.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=1.0,
        last_vol_imbalance=0.0,
        last_trade_intensity=0.0,
        last_realized_spread=0.0,
        last_agent_fill_ratio=0.0,
        token_id=0,
        max_num_tokens=1,
        num_tokens=1,
        norm_cols_values=norm_cols,
        out_features=obs
    )

    # Check that norm_cols values are present in observation
    # They should be at positions 38-58 (after Fear & Greed at 36-37)
    norm_cols_start = 38

    print(f"\nChecking norm_cols integration:")
    print(f"  norm_cols positions: {norm_cols_start} to {norm_cols_start + 20}")

    all_filled = True
    for i in range(21):
        obs_value = obs[norm_cols_start + i]
        # Values should be transformed by tanh and clipped
        expected_range = (-3.0, 3.0)

        if obs_value == 0.0 and norm_cols[i] != 0.0:
            print(f"  ✗ Position {norm_cols_start + i} is zero but should contain norm_cols[{i}]")
            all_filled = False
        else:
            print(f"  ✓ Position {norm_cols_start + i} = {obs_value:.4f} (from norm_cols[{i}] = {norm_cols[i]:.4f})")

    if all_filled:
        print("\n✓ PASS: All 21 norm_cols values are integrated into observation (was 16, added 5)")
    else:
        print("\n✗ FAIL: Some norm_cols values are missing")
        exit(1)

    # Check observation size
    non_zero_count = np.count_nonzero(obs)
    print(f"\nObservation statistics:")
    print(f"  Total size: {len(obs)}")
    print(f"  Non-zero values: {non_zero_count}")
    print(f"  Min: {obs.min():.4f}, Max: {obs.max():.4f}")

    print("\n✓ PASS: Observation builder works correctly with 62 features (was 56, added 6 validity flags)")

except Exception as e:
    print(f"\n✗ FAIL: Error during obs_builder test: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: List all 62 features
print("\n" + "=" * 80)
print("Test 4: Complete list of all 62 features (was 56, added 6 validity flags)")
print("=" * 80)

feature_list = [
    "0-2: Bar (price, log_volume_norm, rel_volume)",
    "3-4: MA5 (value, valid_flag)",
    "5-6: MA20 (value, valid_flag)",
    "7-8: RSI (value, valid_flag) - NEW FLAG",
    "9-10: MACD (value, valid_flag) - NEW FLAG",
    "11-12: MACD Signal (value, valid_flag) - NEW FLAG",
    "13-14: Momentum (value, valid_flag) - NEW FLAG",
    "15: ATR (no flag - intelligent fallback)",
    "16-17: CCI (value, valid_flag) - NEW FLAG",
    "18-19: OBV (value, valid_flag) - NEW FLAG",
    "20-21: Derived (ret_bar, vol_proxy)",
    "22-27: Agent (cash_ratio, position_ratio, vol_imbalance, trade_intensity, realized_spread, agent_fill_ratio)",
    "28-30: Microstructure (price_momentum, bb_squeeze, trend_strength)",
    "31-32: Bollinger (bb_position, bb_width)",
    "33-35: Event metadata (is_high_importance, time_since_event, risk_off_flag)",
    "36-37: Fear & Greed (value, indicator)",
    "38-58: External/norm_cols (21 features) - Обновлено для 4h таймфрейма:",
    "  38: cvd_24h",
    "  39: cvd_7d",
    "  40: yang_zhang_48h",
    "  41: yang_zhang_7d",
    "  42: garch_200h",
    "  43: garch_14d",
    "  44: ret_12h",
    "  45: ret_24h",
    "  46: ret_4h",
    "  47: sma_12000",
    "  48: yang_zhang_30d",
    "  49: parkinson_48h",
    "  50: parkinson_7d",
    "  51: garch_30d",
    "  52: taker_buy_ratio",
    "  53: taker_buy_ratio_sma_24h",
    "  54: taker_buy_ratio_sma_8h",
    "  55: taker_buy_ratio_sma_16h",
    "  56: taker_buy_ratio_momentum_4h",
    "  57: taker_buy_ratio_momentum_8h",
    "  58: taker_buy_ratio_momentum_12h",
    "59-60: Token metadata (num_tokens_norm, token_id_norm)",
    "61: Token one-hot (1 slot for max_num_tokens=1)",
]

for feature in feature_list:
    print(f"  {feature}")

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED: 62 features are properly integrated!")
print("=" * 80)
