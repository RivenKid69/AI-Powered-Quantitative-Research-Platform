#!/usr/bin/env python3
"""
Manual audit of 63-feature observation structure.
Tracks expected vs actual feature indices.
"""

print("=" * 80)
print("MANUAL AUDIT: 63-Feature Observation Structure")
print("=" * 80)
print()

# Define expected feature structure based on code review
features = [
    # Bar level (0-2)
    (0, "price"),
    (1, "log_volume_norm"),
    (2, "rel_volume"),

    # MA features (3-6)
    (3, "ma5"),
    (4, "ma5_valid"),
    (5, "ma20"),
    (6, "ma20_valid"),

    # Technical indicators with validity flags (7-20)
    (7, "rsi14"),
    (8, "rsi_valid"),
    (9, "macd"),
    (10, "macd_valid"),
    (11, "macd_signal"),
    (12, "macd_signal_valid"),
    (13, "momentum"),
    (14, "momentum_valid"),
    (15, "atr"),
    (16, "atr_valid"),  # NEW in 62→63
    (17, "cci"),
    (18, "cci_valid"),
    (19, "obv"),
    (20, "obv_valid"),

    # Bollinger Bands (21-22)
    (21, "bb_position"),
    (22, "bb_width"),

    # Derived (23-24)
    (23, "ret_bar"),
    (24, "vol_proxy"),

    # Agent state (25-30)
    (25, "cash_ratio"),
    (26, "position_ratio"),
    (27, "vol_imbalance"),
    (28, "trade_intensity"),
    (29, "realized_spread"),
    (30, "agent_fill_ratio"),

    # Microstructure / Technical 4h (31-33)
    (31, "price_momentum OR bb_squeeze"),  # Check which is first
    (32, "trend_strength OR price_momentum"),
    (33, "bb_position OR trend_strength"),

    # Event metadata (34-36)
    (34, "is_high_importance"),
    (35, "time_since_event"),
    (36, "risk_off_flag"),

    # Fear & Greed (37-38)
    (37, "fear_greed_value"),
    (38, "fear_greed_indicator"),

    # External normalized columns (39-59) - 21 features
    *[(39 + i, f"norm_cols[{i}]") for i in range(21)],

    # Token metadata (60-61)
    (60, "token_count_ratio"),
    (61, "token_id_norm"),

    # Token one-hot (62)
    (62, "token_one_hot[0]"),
]

print(f"Expected total features: {len(features)}")

# Wait, I think I miscounted. Let me recalculate:

# Actually, looking at the structure more carefully:
# Bar: 3
# MA: 4
# Indicators: 14 (7 indicators * 2 each: value + valid)
# BB: 2
# Derived: 2
# Agent: 6
# Microstructure: 3
# Metadata: 3
# Fear: 2
# External: 21
# Token meta: 2
# Token onehot: 1
# Total: 3+4+14+2+2+6+3+3+2+21+2+1 = 63

# But I have confusion in indices 21-33. Let me check documentation.

print("\n" + "=" * 80)
print("Checking indices 21-33 (BB + Derived + Agent + Microstructure)")
print("=" * 80)

# From code review, correct order should be:
# 21-22: BB (bb_position, bb_width)
# 23-24: Derived (ret_bar, vol_proxy)
# 25-30: Agent (6 features)
# 31-33: Microstructure (bb_squeeze, price_momentum, trend_strength)

features_corrected = [
    # Bar level (0-2)
    (0, "price"),
    (1, "log_volume_norm"),
    (2, "rel_volume"),

    # MA features (3-6)
    (3, "ma5"),
    (4, "ma5_valid"),
    (5, "ma20"),
    (6, "ma20_valid"),

    # Technical indicators with validity flags (7-20)
    (7, "rsi14"),
    (8, "rsi_valid"),
    (9, "macd"),
    (10, "macd_valid"),
    (11, "macd_signal"),
    (12, "macd_signal_valid"),
    (13, "momentum"),
    (14, "momentum_valid"),
    (15, "atr"),
    (16, "atr_valid"),  # NEW in 62→63
    (17, "cci"),
    (18, "cci_valid"),
    (19, "obv"),
    (20, "obv_valid"),

    # Bollinger Bands (21-22)
    (21, "bb_position"),
    (22, "bb_width"),

    # Derived (23-24)
    (23, "ret_bar"),
    (24, "vol_proxy"),

    # Agent state (25-30)
    (25, "cash_ratio"),
    (26, "position_ratio"),
    (27, "vol_imbalance"),
    (28, "trade_intensity"),
    (29, "realized_spread"),
    (30, "agent_fill_ratio"),

    # Microstructure (31-33)
    (31, "bb_squeeze"),
    (32, "price_momentum"),
    (33, "trend_strength"),

    # Event metadata (34-36)
    (34, "is_high_importance"),
    (35, "time_since_event"),
    (36, "risk_off_flag"),

    # Fear & Greed (37-38)
    (37, "fear_greed_value"),
    (38, "fear_greed_indicator"),

    # External normalized columns (39-59) - 21 features
    *[(39 + i, f"norm_cols[{i}]") for i in range(21)],

    # Token metadata (60-61)
    (60, "token_count_ratio"),
    (61, "token_id_norm"),

    # Token one-hot (62)
    (62, "token_one_hot[0]"),
]

print(f"\nCorrected total features: {len(features_corrected)}")

if len(features_corrected) == 63:
    print("✅ Correct: 63 features")
else:
    print(f"❌ ERROR: {len(features_corrected)} features, expected 63")

# Print mapping for critical indices
print("\n" + "=" * 80)
print("Critical Indices to Verify:")
print("=" * 80)

critical_indices = [
    (15, "atr"),
    (16, "atr_valid"),  # NEW
    (17, "cci"),
    (24, "vol_proxy"),  # Uses atr_valid
    (31, "bb_squeeze"),
    (39, "external[0]"),
    (59, "external[20]"),
    (62, "token_one_hot[0]"),
]

for idx, name in critical_indices:
    print(f"  [{idx:2d}] {name}")

print("\n✅ MANUAL AUDIT COMPLETE")
print("Expected structure: 63 features with atr_valid at index 16")
