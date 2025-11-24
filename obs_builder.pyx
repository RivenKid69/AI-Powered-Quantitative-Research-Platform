# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cython cimport Py_ssize_t
from libc.math cimport tanh, log1p, isnan, isinf, isfinite


cdef inline float _clipf(double value, double lower, double upper) nogil:
    """
    Clip value to [lower, upper] range with NaN handling.

    CRITICAL: NaN comparisons are always False in C/Cython, so we must check explicitly.
    If value is NaN, we return 0.0 as a safe default to prevent NaN propagation.

    ISSUE #2 - DESIGN NOTE:
        Converting NaN → 0.0 creates semantic ambiguity for the model:
        - "Missing data" (NaN) becomes indistinguishable from "zero value" (0.0)
        - Model cannot learn special handling for missing data
        - Affects external features (cvd, garch, yang_zhang, etc.) without validity flags

        Technical indicators (MA, RSI, BB) have explicit validity flags (ma5_valid, rsi_valid)
        to signal missing/invalid data, but external features do not.

        Future Enhancement: Add validity flags for all 21 external features by:
        1. Returning (value, is_valid) tuple from mediator._get_safe_float()
        2. Expanding observation vector by +21 dims for validity flags
        3. Retraining models to use validity information

        Current behavior is by design to prevent NaN propagation, but is suboptimal.
    """
    if isnan(value):
        return 0.0  # Silent conversion - see ISSUE #2 note above
    if value < lower:
        value = lower
    elif value > upper:
        value = upper
    return <float>value


cdef inline void _validate_price(float price, str param_name) except *:
    """
    Validate that price is finite and positive.

    This function enforces critical data integrity constraints for price data:
    1. Price must not be NaN (indicates missing or corrupted data)
    2. Price must not be Inf/-Inf (indicates computation overflow)
    3. Price must be strictly positive (negative/zero prices are invalid)

    Args:
        price: The price value to validate
        param_name: Name of the parameter for error messages (e.g., "price", "prev_price")

    Raises:
        ValueError: If price fails validation with detailed diagnostic message

    Research references:
    - "Data validation best practices" (Cube Software)
    - "Incomplete Data - Machine Learning Trading" (OMSCS)
    - Financial data standards require positive, finite prices
    """
    if isnan(price):
        raise ValueError(
            f"Invalid {param_name}: NaN (Not a Number). "
            f"This indicates missing or corrupted market data. "
            f"All price inputs must be valid finite numbers. "
            f"Check data source integrity and preprocessing pipeline."
        )

    if isinf(price):
        sign = "positive" if price > 0 else "negative"
        raise ValueError(
            f"Invalid {param_name}: {sign} infinity. "
            f"This indicates arithmetic overflow in upstream calculations. "
            f"All price inputs must be finite values. "
            f"Review data transformations and numerical stability."
        )

    if price <= 0.0:
        raise ValueError(
            f"Invalid {param_name}: {price:.10f}. "
            f"Price must be strictly positive (> 0). "
            f"Negative or zero prices are invalid in trading systems. "
            f"This may indicate data errors, incorrect units, or "
            f"issues with price normalization/denormalization."
        )


cdef inline void _validate_portfolio_value(float value, str param_name) except *:
    """
    Validate portfolio value (cash or units) - finite but can be zero.

    Portfolio values have different validation rules than prices:
    - CAN be 0.0 (valid state: no cash or no position)
    - CAN be negative for cash (short positions, margin debt)
    - CANNOT be NaN (indicates missing/corrupted data)
    - CANNOT be Inf (indicates calculation overflow)

    Args:
        value: The portfolio value to validate (cash or units)
        param_name: Parameter name for error messages

    Raises:
        ValueError: If value is NaN or Inf

    Best practices:
    - Zero cash/units is valid portfolio state
    - Negative cash can be valid (margin, short positions)
    - NaN/Inf indicate data corruption and must be caught

    References:
    - "Investment Model Validation" (CFA Institute)
    - "Best Practices for Ensuring Financial Data Accuracy" (Paystand)
    """
    if isnan(value):
        raise ValueError(
            f"Invalid {param_name}: NaN (Not a Number). "
            f"Portfolio values must be finite numbers. "
            f"NaN indicates missing or corrupted portfolio state. "
            f"Check state management and data pipeline integrity."
        )

    if isinf(value):
        sign = "positive" if value > 0 else "negative"
        raise ValueError(
            f"Invalid {param_name}: {sign} infinity. "
            f"Portfolio values must be finite. "
            f"Infinity indicates arithmetic overflow in calculations. "
            f"Check portfolio valuation logic and numerical stability."
        )


cdef inline void _validate_volume_metric(float value, str param_name) except *:
    """
    Validate volume-derived metrics (log_volume_norm, rel_volume).

    Volume metrics are derived from market data transformations and must be finite.
    Unlike prices, they CAN be zero (no volume) or negative (theoretical edge case).
    However, they CANNOT be NaN or Inf as this indicates upstream calculation errors.

    Typical range: [-1, 1] due to tanh normalization, but we validate finitude only.

    Args:
        value: The volume metric to validate
        param_name: Parameter name for error messages (e.g., "log_volume_norm")

    Raises:
        ValueError: If value is NaN or Inf with detailed diagnostic message

    Research references:
    - "Defense in Depth" (OWASP): Multiple validation layers prevent NaN propagation
    - "Data Validation Best Practices" (Cube Software): Validate at boundaries
    - "Fail-fast validation" (Martin Fowler): Catch errors early in pipeline

    Design rationale:
    - Volume metrics computed in mediator._extract_market_data()
    - Formula: tanh(log1p(volume / normalizer)) with volume >= 0 guaranteed by P0
    - P0 layer (_get_safe_float with min_value=0.0) prevents negative volumes
    - With volume >= 0, tanh(log1p(...)) always yields finite result in [-1, 1]
    - This P2 validation catches any remaining edge cases or pipeline errors
    - Fail-fast approach prevents silent NaN propagation to observation vector
    """
    if isnan(value):
        raise ValueError(
            f"Invalid {param_name}: NaN (Not a Number). "
            f"Volume metrics must be finite numbers. "
            f"NaN indicates corrupted market data or calculation error. "
            f"Check volume data source and normalization pipeline. "
            f"Common causes: missing volume data, division by zero, log of negative."
        )

    if isinf(value):
        sign = "positive" if value > 0 else "negative"
        raise ValueError(
            f"Invalid {param_name}: {sign} infinity. "
            f"Volume metrics must be finite. "
            f"Infinity indicates numerical overflow in volume normalization. "
            f"Check volume scaling factors and log1p/tanh computations. "
            f"Review mediator.py _extract_market_data() for calculation errors."
        )


cpdef int compute_n_features(list layout):
    """Utility used by legacy Python code to count feature slots."""
    cdef int total = 0
    cdef dict block
    for block in layout:
        total += <int>block.get("size", 0)
    return total


cdef void build_observation_vector_c(
    float price,
    float prev_price,
    float log_volume_norm,
    float rel_volume,
    float ma5,
    float ma20,
    float rsi14,
    float macd,
    float macd_signal,
    float momentum,
    float atr,
    float cci,
    float obv,
    float bb_lower,
    float bb_upper,
    float is_high_importance,
    float time_since_event,
    float fear_greed_value,
    bint has_fear_greed,
    bint risk_off_flag,
    float cash,
    float units,
    float signal_pos,
    float last_vol_imbalance,
    float last_trade_intensity,
    float last_realized_spread,
    float last_agent_fill_ratio,
    int token_id,
    int max_num_tokens,
    int num_tokens,
    float[::1] norm_cols_values,
    unsigned char[::1] norm_cols_validity,
    bint enable_validity_flags,
    float[::1] out_features
) noexcept nogil:
    """Populate ``out_features`` with the observation vector without acquiring the GIL."""

    cdef int feature_idx = 0
    cdef float feature_val
    cdef float indicator
    cdef double price_d = price
    cdef double prev_price_d = prev_price
    cdef double position_value
    cdef double total_worth
    cdef double ret_bar
    cdef double vol_proxy
    cdef double price_momentum
    cdef double bb_squeeze
    cdef double trend_strength
    cdef double bb_width
    cdef bint ma5_valid
    cdef bint ma20_valid
    cdef bint rsi_valid
    cdef bint macd_valid
    cdef bint macd_signal_valid
    cdef bint momentum_valid
    cdef bint atr_valid
    cdef bint cci_valid
    cdef bint obv_valid
    cdef bint bb_valid
    cdef double min_bb_width
    cdef int padded_tokens
    cdef Py_ssize_t i

    # --- Bar level block ---------------------------------------------------
    out_features[feature_idx] = price
    feature_idx += 1
    out_features[feature_idx] = log_volume_norm
    feature_idx += 1
    out_features[feature_idx] = rel_volume
    feature_idx += 1

    ma5_valid = not isnan(ma5)
    out_features[feature_idx] = ma5 if ma5_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if ma5_valid else 0.0
    feature_idx += 1

    ma20_valid = not isnan(ma20)
    out_features[feature_idx] = ma20 if ma20_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if ma20_valid else 0.0
    feature_idx += 1

    # Technical indicators with NaN handling (early bars may not have enough history)
    # RSI with validity flag
    # CRITICAL: RSI requires ~14 bars for first valid value
    # Fallback 50.0 creates AMBIGUITY: neutral RSI (50) vs insufficient data (50)
    # Validity flag eliminates this: model can distinguish valid neutral from missing data
    rsi_valid = not isnan(rsi14)
    out_features[feature_idx] = rsi14 if rsi_valid else 50.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if rsi_valid else 0.0
    feature_idx += 1

    # MACD with validity flag
    # CRITICAL: MACD requires ~26 bars for first valid value (12+26 EMA periods)
    # Fallback 0.0 creates AMBIGUITY: no divergence (0) vs insufficient data (0)
    # Validity flag eliminates this: model can distinguish valid zero from missing data
    macd_valid = not isnan(macd)
    out_features[feature_idx] = macd if macd_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if macd_valid else 0.0
    feature_idx += 1

    # MACD Signal with validity flag
    # CRITICAL: MACD Signal requires ~35 bars (26 for MACD + 9 for signal line)
    # Fallback 0.0 creates AMBIGUITY: no signal (0) vs insufficient data (0)
    # Validity flag eliminates this: model can distinguish valid zero from missing data
    macd_signal_valid = not isnan(macd_signal)
    out_features[feature_idx] = macd_signal if macd_signal_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if macd_signal_valid else 0.0
    feature_idx += 1

    # Momentum with validity flag
    # CRITICAL: Momentum requires ~10 bars for first valid value
    # Fallback 0.0 creates AMBIGUITY: no price movement (0) vs insufficient data (0)
    # Validity flag eliminates this: model can distinguish valid zero from missing data
    momentum_valid = not isnan(momentum)
    out_features[feature_idx] = momentum if momentum_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if momentum_valid else 0.0
    feature_idx += 1

    # ATR with validity flag
    # CRITICAL: ATR requires ~14 bars for first valid value (Wilder's smoothing EMA_14)
    # Fallback price*0.01 (1%) creates AMBIGUITY: calm market (1%) vs insufficient data (1%)
    # Validity flag eliminates this: model can distinguish real low volatility from missing data
    # IMPORTANT: This flag is used by vol_proxy calculation to prevent NaN propagation
    atr_valid = not isnan(atr)
    out_features[feature_idx] = atr if atr_valid else <float>(price_d * 0.01)
    feature_idx += 1
    out_features[feature_idx] = 1.0 if atr_valid else 0.0
    feature_idx += 1

    # CCI with validity flag
    # CRITICAL: CCI requires ~20 bars for first valid value
    # Fallback 0.0 creates AMBIGUITY: at average level (0) vs insufficient data (0)
    # Validity flag eliminates this: model can distinguish valid zero from missing data
    cci_valid = not isnan(cci)
    out_features[feature_idx] = cci if cci_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if cci_valid else 0.0
    feature_idx += 1

    # OBV with validity flag
    # CRITICAL: OBV requires only 1 bar for first valid value
    # Fallback 0.0 creates AMBIGUITY: volume balance (0) vs insufficient data (0)
    # Validity flag eliminates this: model can distinguish valid zero from missing data
    obv_valid = not isnan(obv)
    out_features[feature_idx] = obv if obv_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if obv_valid else 0.0
    feature_idx += 1

    # CRITICAL: Derived price/volatility signals (bar-to-bar return for current timeframe)
    # ret_bar calculation (feature index 22, was 14 in 56-feature, 20 in 62-feature):
    # - Formula: tanh((price_d - prev_price_d) / (prev_price_d + 1e-8))
    # - Numerator: price_d - prev_price_d (price change)
    # - Denominator: prev_price_d + 1e-8 (epsilon prevents division by zero)
    # - tanh normalization: maps (-inf, +inf) → (-1, 1)
    #
    # Safety guarantees:
    # 1. Division by zero: Impossible due to +1e-8 epsilon
    #    Even if prev_price_d = 0.0: division = x / 1e-8 = large finite number
    #
    # 2. NaN/Inf protection: Enforced by fail-fast validation at entry points
    #    - P0: Mediator validation (_validate_critical_price at mediator.py:1015)
    #    - P1: Wrapper validation (_validate_price at obs_builder.pyx:469-470)
    #    Both price AND prev_price are validated as finite, positive, non-zero
    #    If validation fails → ValueError raised immediately (fail-fast)
    #
    # 3. No silent failures: Invalid data causes immediate exception, not silent corruption
    #
    # Direct call path (lob_state_cython.pyx:62):
    # - Only used for feature vector size calculation with dummy zeros
    # - price=0.0, prev_price=0.0 → ret_bar = tanh(0/1e-8) = tanh(0) = 0.0
    # - Safe and correct for initialization purposes
    #
    # Design philosophy: Fail-fast at entry (P0/P1) > Silent fallbacks in computation
    # Research references:
    # - "Fail-fast validation" (Martin Fowler): Catch errors early, fail loudly
    # - IEEE 754: NaN propagation requires explicit handling at data boundaries
    # - Financial data standards: Validation at ingestion, not in calculations
    ret_bar = tanh((price_d - prev_price_d) / (prev_price_d + 1e-8))
    out_features[feature_idx] = <float>ret_bar
    feature_idx += 1

    # vol_proxy calculation with ATR validity check to prevent NaN propagation
    # CRITICAL: Must use atr_valid flag to prevent NaN when ATR is unavailable (first ~14 bars)
    # Without this check, vol_proxy becomes NaN during warmup period, violating
    # the core guarantee of "no NaN in observation vector"
    #
    # Research references:
    # - IEEE 754: NaN propagates through arithmetic (log1p(NaN) = NaN, tanh(NaN) = NaN)
    # - "Defense in Depth" (OWASP): Validate before use, not just at storage
    # - Wilder (1978): ATR requires 14 bars minimum (EMA smoothing)
    if atr_valid:
        vol_proxy = tanh(log1p(atr / (price_d + 1e-8)))
    else:
        # Use fallback ATR value (1% of price) for vol_proxy calculation
        # This ensures vol_proxy is always finite, even during warmup
        atr_fallback = price_d * 0.01
        vol_proxy = tanh(log1p(atr_fallback / (price_d + 1e-8)))
    out_features[feature_idx] = <float>vol_proxy
    feature_idx += 1

    # --- Agent state block -------------------------------------------------
    position_value = units * price_d
    total_worth = cash + position_value

    if total_worth <= 1e-8:
        feature_val = 1.0
    else:
        feature_val = _clipf(cash / total_worth, 0.0, 1.0)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    if total_worth <= 1e-8:
        feature_val = 0.0
    else:
        feature_val = <float>tanh(position_value / (total_worth + 1e-8))
    out_features[feature_idx] = feature_val
    feature_idx += 1

    out_features[feature_idx] = <float>tanh(last_vol_imbalance)
    feature_idx += 1
    out_features[feature_idx] = <float>tanh(last_trade_intensity)
    feature_idx += 1

    feature_val = _clipf(last_realized_spread, -0.1, 0.1)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    out_features[feature_idx] = last_agent_fill_ratio
    feature_idx += 1

    # FIX (2025-11-24): Add signal_pos to observation vector
    # CRITICAL: In signal_only mode, model needs to know its target position
    # signal_pos is the TARGET position in [-1, 1] (long-only: [0, 1])
    # This enables the model to make informed decisions about position changes
    out_features[feature_idx] = _clipf(signal_pos, -1.0, 1.0)
    feature_idx += 1

    # --- Technical indicators for 4h timeframe (replaces microstructure) ---
    # Microstructure features (ofi_proxy, qimb, micro_dev) are not applicable for 4h timeframe
    # as they require high-frequency order flow data. Replaced with candlestick-based indicators.

    # 1. Price momentum (replaces ofi_proxy) - captures trend direction and strength
    # Uses normalized momentum indicator to measure price movement strength
    # Normalized by 1% of price (price_d * 0.01) for sensitivity to typical intraday moves
    # Validity flag: Uses momentum_valid flag to check data availability
    # If momentum is invalid (first 10 bars), use 0.0 (no momentum)
    if momentum_valid:
        price_momentum = tanh(momentum / (price_d * 0.01 + 1e-8))
    else:
        price_momentum = 0.0
    out_features[feature_idx] = <float>price_momentum
    feature_idx += 1

    # 2. Bollinger Bands squeeze (replaces qimb) - measures volatility regime
    # High value = high volatility (wide bands), low value = low volatility (squeeze)
    # Normalized by full price (price_d) not 1% because bb_width is typically 1-5% of price
    # This ensures the normalized value is in a reasonable range for tanh
    # NaN handling: if BB not ready (first 20 bars), use 0.0 (neutral volatility)
    #
    # CRITICAL: Validate BOTH bb_lower AND bb_upper for completeness
    # - Bollinger Bands require both bounds to be valid
    # - If only one is NaN/Inf, derived features (bb_width, bb_position) become NaN
    # - bb_width = bb_upper - bb_lower: if either is NaN/Inf → NaN propagation
    # - Must check finitude (not just NaN) to catch Inf values
    # - Logical consistency: bb_upper should be >= bb_lower (sanity check)
    #
    # Research references:
    # - "Bollinger Bands" (John Bollinger): Upper band > Lower band by definition
    # - "Defense in Depth" (OWASP): Validate all required inputs, not just subset
    # - "Data Validation Best Practices": Complete validation prevents partial failures
    bb_valid = (not isnan(bb_lower) and not isnan(bb_upper) and
                isfinite(bb_lower) and isfinite(bb_upper) and
                bb_upper >= bb_lower)
    if bb_valid:
        bb_squeeze = tanh((bb_upper - bb_lower) / (price_d + 1e-8))
    else:
        bb_squeeze = 0.0
    out_features[feature_idx] = <float>bb_squeeze
    feature_idx += 1

    # 3. Trend strength via MACD divergence (replaces micro_dev) - measures trend strength
    # Positive = bullish trend, negative = bearish trend, magnitude = strength
    # Normalized by 1% of price (price_d * 0.01) similar to price_momentum for consistency
    # Validity flags: Uses macd_valid and macd_signal_valid flags to check data availability
    # If either MACD or signal is invalid (first ~26-35 bars), use 0.0 (no trend signal)
    if macd_valid and macd_signal_valid:
        trend_strength = tanh((macd - macd_signal) / (price_d * 0.01 + 1e-8))
    else:
        trend_strength = 0.0
    out_features[feature_idx] = <float>trend_strength
    feature_idx += 1

    # --- Bollinger band context -------------------------------------------
    # Position within bands and band width - critical features for volatility-based strategies
    # NOTE: bb_valid is already computed above for bb_squeeze with FULL validation:
    #       - Validates both bb_lower AND bb_upper are finite (not NaN/Inf)
    #       - Ensures logical consistency: bb_upper >= bb_lower
    #       - If validation fails, both features default to safe values
    #
    # Defense-in-depth: Double-check bb_width calculation
    # Even with bb_valid check, explicitly verify bb_width is finite
    # This catches any remaining edge cases or calculation errors
    bb_width = bb_upper - bb_lower
    min_bb_width = price_d * 0.0001

    # Feature 1: Price position within Bollinger Bands
    # Default: 0.5 = at the middle (when bands not available)
    # Standard: 0.0 = at lower band, 1.0 = at upper band
    #
    # FIX CRITICAL BUG (2025-11-23): Changed asymmetric [-1.0, 2.0] to symmetric [-1.0, 1.0]
    # =================================================================================
    # PROBLEM with old asymmetric range [-1.0, 2.0]:
    # - Creates training distribution bias: model sees +2.0 (bullish extreme) but NEVER -2.0
    # - Neural networks prefer symmetric, zero-centered inputs (Goodfellow et al. 2016)
    # - Batch normalization and tanh activation work best with symmetric data (Ioffe & Szegedy 2015)
    # - Market asymmetry (crypto pumps > dumps) should be learned from DATA, not imposed by features
    #
    # OLD RATIONALE (now deprecated):
    # - "Captures crypto-specific behavior" - TRUE, but WRONG approach
    # - Market microstructure asymmetry should come from RAW price movements
    # - Feature engineering should remain UNBIASED - let model learn asymmetry
    #
    # NEW APPROACH (symmetric [-1.0, 1.0]):
    # - Unbiased feature normalization following ML best practices
    # - Model can still learn crypto asymmetry from actual price behavior
    # - Better convergence due to symmetric input distribution
    # - Consistent with other normalized features (most use [-1, 1] or [0, 1])
    #
    # Research support:
    # - Goodfellow et al. (2016): "Deep Learning" - inputs should be zero-centered and symmetric
    # - Ioffe & Szegedy (2015): "Batch Normalization" - symmetric distributions improve convergence
    # - Lopez de Prado (2018): "Advances in Financial ML" - feature engineering should be unbiased
    # - Makarov & Schoar (2020): Crypto asymmetry is DATA property, not FEATURE property
    #
    # Examples with new range:
    # - Price at upper band + 1*width → bb_position = 1.0 (extreme bullish, clipped)
    # - Price at lower band - 1*width → bb_position = -1.0 (extreme bearish, clipped)
    # - Price at middle → bb_position = 0.5 (neutral)
    # - Price at upper band → bb_position = 1.0
    # - Price at lower band → bb_position = 0.0
    # =================================================================================
    #
    # Defense-in-depth validation:
    # 1. Primary: bb_valid check (both bands finite and consistent)
    # 2. Secondary: bb_width > min_bb_width (avoid division by near-zero)
    # 3. Tertiary: _clipf handles any remaining NaN via NaN-to-zero conversion
    #
    # This triple-layer approach ensures bb_position CANNOT be NaN:
    # - Layer 1: Prevents invalid inputs from being used
    # - Layer 2: Prevents division by zero/near-zero
    # - Layer 3: Final safety net converts any NaN to 0.0
    if (not bb_valid) or bb_width <= min_bb_width:
        feature_val = 0.5
    else:
        # Additional safety: verify bb_width is finite before division
        if not isfinite(bb_width):
            feature_val = 0.5
        else:
            # FIX: Symmetric clip [-1.0, 1.0] for unbiased neural network training
            feature_val = _clipf((price_d - bb_lower) / (bb_width + 1e-9), -1.0, 1.0)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    # Feature 2: Normalized band width (volatility measure)
    # 0.0 = bands not available or zero width
    #
    # Defense-in-depth validation:
    # 1. Primary: bb_valid check ensures inputs are finite
    # 2. Secondary: Verify bb_width is finite before normalization
    # 3. Tertiary: _clipf handles any remaining NaN
    if bb_valid:
        # Additional safety: verify bb_width is finite
        if not isfinite(bb_width):
            feature_val = 0.0
        else:
            feature_val = _clipf(bb_width / (price_d + 1e-8), 0.0, 10.0)
    else:
        feature_val = 0.0
    out_features[feature_idx] = feature_val
    feature_idx += 1

    # --- Event metadata ----------------------------------------------------
    out_features[feature_idx] = is_high_importance
    feature_idx += 1

    out_features[feature_idx] = <float>tanh(time_since_event / 24.0)
    feature_idx += 1

    out_features[feature_idx] = 1.0 if risk_off_flag else 0.0
    feature_idx += 1

    # --- Fear & Greed ------------------------------------------------------
    if has_fear_greed:
        feature_val = _clipf(fear_greed_value / 100.0, -3.0, 3.0)
        indicator = 1.0
    else:
        feature_val = 0.0
        indicator = 0.0
    out_features[feature_idx] = feature_val
    feature_idx += 1
    out_features[feature_idx] = indicator
    feature_idx += 1

    # --- External normalised columns --------------------------------------
    # Process 21 external features (cvd, garch, yang_zhang, returns, taker_buy_ratio, etc.)
    # ISSUE #2 FIX (Phase 2 - COMPLETE): Now write validity flags to observation vector
    # to enable model to distinguish missing data (NaN) from zero values.
    # Validity flags are written AFTER token metadata (following FEATURES_LAYOUT order).
    for i in range(norm_cols_values.shape[0]):
        # Apply tanh normalization first, then clip to safe range
        # Note: _clipf converts NaN→0.0 (see ISSUE #2 in _clipf docstring)
        feature_val = _clipf(tanh(norm_cols_values[i]), -3.0, 3.0)
        out_features[feature_idx] = feature_val
        feature_idx += 1

    # --- Token metadata ----------------------------------------------------
    if max_num_tokens > 0:
        # Normalised statistics to keep vector length fixed
        feature_val = _clipf(num_tokens / (<double>max_num_tokens), 0.0, 1.0)
        out_features[feature_idx] = feature_val
        feature_idx += 1

        if 0 <= token_id < max_num_tokens:
            feature_val = _clipf(token_id / (<double>max_num_tokens), 0.0, 1.0)
        else:
            feature_val = 0.0
        out_features[feature_idx] = feature_val
        feature_idx += 1

        padded_tokens = max_num_tokens
        for i in range(padded_tokens):
            out_features[feature_idx + i] = 0.0

        if 0 <= token_id < num_tokens and token_id < max_num_tokens:
            out_features[feature_idx + token_id] = 1.0

        feature_idx += padded_tokens

    # --- External validity flags (NEW - Phase 2 of ISSUE #2 fix) ----------
    # Write validity flags for external features to enable model to distinguish
    # missing data (NaN) from zero values. This eliminates semantic ambiguity.
    # Position: After token metadata block (indices 63-83 for 21 external features)
    if enable_validity_flags:
        for i in range(norm_cols_values.shape[0]):
            out_features[feature_idx] = 1.0 if norm_cols_validity[i] else 0.0
            feature_idx += 1


cpdef void build_observation_vector(
    float price,
    float prev_price,
    float log_volume_norm,
    float rel_volume,
    float ma5,
    float ma20,
    float rsi14,
    float macd,
    float macd_signal,
    float momentum,
    float atr,
    float cci,
    float obv,
    float bb_lower,
    float bb_upper,
    float is_high_importance,
    float time_since_event,
    float fear_greed_value,
    bint has_fear_greed,
    bint risk_off_flag,
    float cash,
    float units,
    float signal_pos,
    float last_vol_imbalance,
    float last_trade_intensity,
    float last_realized_spread,
    float last_agent_fill_ratio,
    int token_id,
    int max_num_tokens,
    int num_tokens,
    float[::1] norm_cols_values,
    unsigned char[::1] norm_cols_validity,
    bint enable_validity_flags,
    float[::1] out_features
):
    """
    Python-callable wrapper that forwards to the ``nogil`` implementation.

    CRITICAL: Validates critical inputs before processing to prevent NaN/Inf propagation.
    This is the entry point for all observation vector construction and must enforce
    data integrity constraints.

    Validation performed:
    - price must be finite (not NaN/Inf) and positive (> 0)
    - prev_price must be finite (not NaN/Inf) and positive (> 0)
    - log_volume_norm must be finite (not NaN/Inf), can be 0 or negative
    - rel_volume must be finite (not NaN/Inf), can be 0 or negative
    - cash must be finite (not NaN/Inf), can be 0 or negative
    - units must be finite (not NaN/Inf), can be 0 or negative

    If validation fails, ValueError is raised with diagnostic information.
    This fail-fast approach prevents silent data corruption in the observation vector.

    Best practices implemented:
    - Price validation: Strict (must be > 0)
    - Volume metrics validation: Must be finite (can be 0 or negative)
    - Portfolio validation: Allows 0/negative but not NaN/Inf
    - Fail-fast approach catches data issues early
    - Clear error messages for debugging
    """
    # CRITICAL: Validate price inputs before any computation
    # This prevents NaN/Inf propagation through 15+ calculations downstream
    _validate_price(price, "price")
    _validate_price(prev_price, "prev_price")

    # CRITICAL: Validate volume metrics to prevent NaN propagation
    # These are computed from raw volume data and can be corrupted upstream
    # Without this check, corrupted values would be written directly to observation array
    _validate_volume_metric(log_volume_norm, "log_volume_norm")
    _validate_volume_metric(rel_volume, "rel_volume")

    # Validate portfolio state (cash and units)
    # These can be 0 or negative (valid states) but not NaN/Inf
    _validate_portfolio_value(cash, "cash")
    _validate_portfolio_value(units, "units")

    build_observation_vector_c(
        price,
        prev_price,
        log_volume_norm,
        rel_volume,
        ma5,
        ma20,
        rsi14,
        macd,
        macd_signal,
        momentum,
        atr,
        cci,
        obv,
        bb_lower,
        bb_upper,
        is_high_importance,
        time_since_event,
        fear_greed_value,
        has_fear_greed,
        risk_off_flag,
        cash,
        units,
        signal_pos,
        last_vol_imbalance,
        last_trade_intensity,
        last_realized_spread,
        last_agent_fill_ratio,
        token_id,
        max_num_tokens,
        num_tokens,
        norm_cols_values,
        norm_cols_validity,
        enable_validity_flags,
        out_features,
    )
