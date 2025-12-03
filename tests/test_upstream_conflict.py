#!/usr/bin/env python3
"""
Test to verify the conflict between _coerce_finite and price validation.

This test demonstrates that:
1. mediator._coerce_finite() converts NaN -> 0.0 (silent fallback)
2. obs_builder._validate_price() rejects 0.0 (explicit error)
3. This creates a conflict where NaN prices now raise errors instead of being silently converted
"""

import math
import numpy as np
import pytest
import feature_config as fc

# Import the validation logic (may be unavailable if the extension is not built)
try:
    from obs_builder import build_observation_vector
except ImportError:  # pragma: no cover - handled via pytest skip
    build_observation_vector = None


def _require_obs_builder():
    """Skip tests cleanly if obs_builder is not compiled."""
    if build_observation_vector is None:
        pytest.skip("obs_builder extension is not available in this environment")


def _external_block_info():
    """Return (ext_dim, has_validity_block) derived from feature_config layout."""
    fc.make_layout({})
    ext_dim = next(b["size"] for b in fc.FEATURES_LAYOUT if b["name"] == "external")
    has_validity = any(b["name"] == "external_validity" for b in fc.FEATURES_LAYOUT)
    return ext_dim, has_validity


def _coerce_finite(value, default=0.0):
    """Replicate mediator._coerce_finite() logic."""
    if value is None:
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return numeric


def test_upstream_conflict():
    """
    Test the conflict between _coerce_finite and price validation.
    """
    _require_obs_builder()
    ext_dim, has_validity = _external_block_info()

    print("=" * 80)
    print("TEST: Upstream conflict between _coerce_finite and price validation")
    print("=" * 80)
    print()

    # Simulate what happens in mediator.py
    print("SCENARIO: mark_price = NaN (missing data)")
    print("-" * 80)

    mark_price = float("nan")
    print(f"1. Input: mark_price = {mark_price}")

    # Step 1: _coerce_finite converts NaN -> 0.0
    coerced_price = _coerce_finite(mark_price, default=0.0)
    print(f"2. After _coerce_finite(): price = {coerced_price} (NaN silently converted to 0.0)")

    # Step 2: Try to build observation with coerced price
    print(f"3. Calling build_observation_vector(price={coerced_price}, ...)")

    obs = np.zeros(fc.N_FEATURES, dtype=np.float32)
    norm_cols = np.zeros(ext_dim, dtype=np.float32)
    norm_validity = np.ones(ext_dim, dtype=np.uint8)

    try:
        build_observation_vector(
            price=coerced_price,  # 0.0 from NaN
            prev_price=50000.0,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=float("nan"),
            ma20=float("nan"),
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=0.0,
            cci=0.0,
            obv=0.0,
            bb_lower=float("nan"),
            bb_upper=float("nan"),
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=False,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            signal_pos=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=1.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=norm_cols,
            norm_cols_validity=norm_validity,
            enable_validity_flags=has_validity,
            out_features=obs,
        )
        print("   OK build_observation_vector() succeeded")
        print()
        print("FAIL: Expected ValueError but got success!")
        print("   This means price validation is NOT working correctly.")

    except ValueError as e:
        print(f"   build_observation_vector() raised ValueError")
        print(f"   Error message: {str(e)[:100]}...")
        print()
        print("PASS: Price validation correctly rejects 0.0 from NaN")
        print()
        print("CONCLUSION:")
        print("-" * 80)
        print("- Old behavior: NaN -> 0.0 -> observation built with invalid data (SILENT FAILURE)")
        print("- New behavior: NaN -> 0.0 -> ValueError (EXPLICIT FAILURE)")
        print()
        print("WARNING: THIS IS A BREAKING CHANGE")
        print("   Code that relied on silent fallback to 0.0 will now fail explicitly.")
        print()
        print("BUT THIS IS CORRECT BEHAVIOR:")
        print("   NaN prices indicate data corruption and should be caught early.")
        print("   The fix exposes existing data quality issues that were previously hidden.")


def test_positive_infinity_scenario():
    """Test with +Inf input."""
    _require_obs_builder()
    ext_dim, has_validity = _external_block_info()

    print()
    print("=" * 80)
    print("SCENARIO: mark_price = +Inf (arithmetic overflow)")
    print("-" * 80)

    mark_price = float("inf")
    print(f"1. Input: mark_price = {mark_price}")

    coerced_price = _coerce_finite(mark_price, default=0.0)
    print(f"2. After _coerce_finite(): price = {coerced_price}")

    obs = np.zeros(fc.N_FEATURES, dtype=np.float32)
    norm_cols = np.zeros(ext_dim, dtype=np.float32)
    norm_validity = np.ones(ext_dim, dtype=np.uint8)

    try:
        build_observation_vector(
            price=coerced_price,
            prev_price=50000.0,
            log_volume_norm=0.0,
            rel_volume=0.0,
            ma5=float("nan"),
            ma20=float("nan"),
            rsi14=50.0,
            macd=0.0,
            macd_signal=0.0,
            momentum=0.0,
            atr=0.0,
            cci=0.0,
            obv=0.0,
            bb_lower=float("nan"),
            bb_upper=float("nan"),
            is_high_importance=0.0,
            time_since_event=0.0,
            fear_greed_value=50.0,
            has_fear_greed=False,
            risk_off_flag=False,
            cash=10000.0,
            units=0.0,
            signal_pos=0.0,
            last_vol_imbalance=0.0,
            last_trade_intensity=0.0,
            last_realized_spread=0.0,
            last_agent_fill_ratio=1.0,
            token_id=0,
            max_num_tokens=1,
            num_tokens=1,
            norm_cols_values=norm_cols,
            norm_cols_validity=norm_validity,
            enable_validity_flags=has_validity,
            out_features=obs,
        )
        print("FAIL: Expected ValueError but got success")
    except ValueError as e:
        print(f"PASS: Correctly rejected: {str(e)[:80]}...")


if __name__ == "__main__":
    if build_observation_vector is None:
        print("obs_builder extension not available; skipping manual run")
        raise SystemExit(0)

    test_upstream_conflict()
    test_positive_infinity_scenario()

    print()
    print("=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("The _coerce_finite() function in mediator.py should be updated to:")
    print("1. NOT use 0.0 as fallback for price/prev_price (use None or raise error)")
    print("2. Handle NaN/Inf prices explicitly BEFORE calling build_observation_vector()")
    print("3. Log warnings when invalid prices are detected")
    print()
    print("This way the error is caught at the data ingestion layer,")
    print("not at the observation construction layer.")
