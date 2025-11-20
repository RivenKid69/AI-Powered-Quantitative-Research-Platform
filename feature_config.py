# Default normalization constants
DEFAULT_TANH_CLIP = 0.999
OBS_EPS = 1e-8
# Default maximum sizes
MAX_NUM_TOKENS = 1  # Changed from 16 to match mediator and lob_state_cython
EXT_NORM_DIM = 21  # Expanded from 16 to 21 to include all technical features (including taker_buy_ratio derivatives)

# Initialize feature layout (to be populated by make_layout)
FEATURES_LAYOUT = []

def make_layout(obs_params=None):
    """Build the observation feature layout list based on parameters."""
    global FEATURES_LAYOUT, N_FEATURES
    if obs_params is None:
        obs_params = {}
    # Determine actual dimensions from parameters or defaults
    max_tokens = obs_params.get('max_num_tokens', MAX_NUM_TOKENS)
    ext_dim = obs_params.get('ext_norm_dim', EXT_NORM_DIM)
    include_fear = obs_params.get('include_fear_greed', obs_params.get('use_dynamic_risk', False))
    # Define feature blocks
    # IMPORTANT: This order MUST match obs_builder.pyx build_observation_vector_c() implementation!
    # See obs_builder.pyx:236-590 for the exact order of feature construction.
    layout = []

    # Block 1: Bar-level features (indices 0-2)
    layout.append({
        "name": "bar",
        "size": 3,
        "dtype": "float32",
        "clip": None,
        "scale": 1.0,
        "bias": 0.0,
        "source": "bars",
        "description": "price, log_volume_norm, rel_volume"
    })

    # Block 2: MA5 (indices 3-4)
    layout.append({
        "name": "ma5",
        "size": 2,
        "dtype": "float32",
        "clip": None,
        "scale": 1.0,
        "bias": 0.0,
        "source": "indicators",
        "description": "ma5, is_ma5_valid"
    })

    # Block 3: MA20 (indices 5-6)
    layout.append({
        "name": "ma20",
        "size": 2,
        "dtype": "float32",
        "clip": None,
        "scale": 1.0,
        "bias": 0.0,
        "source": "indicators",
        "description": "ma20, is_ma20_valid"
    })

    # Block 4: Technical indicators with validity flags (indices 7-20)
    # Includes: rsi14, macd, macd_signal, momentum, atr, cci, obv + validity flags
    # Total: 7 indicators × 2 (value + flag) = 14 features
    layout.append({
        "name": "indicators",
        "size": 14,
        "dtype": "float32",
        "clip": None,
        "scale": 1.0,
        "bias": 0.0,
        "source": "indicators",
        "description": "rsi14, macd, macd_signal, momentum, atr, cci, obv (each with validity flag)"
    })

    # Block 5: Derived price/volatility signals (indices 21-22)
    # NOTE: This comes AFTER indicators, not before! (Fixed from previous incorrect ordering)
    layout.append({
        "name": "derived",
        "size": 2,
        "dtype": "float32",
        "clip": DEFAULT_TANH_CLIP,
        "scale": 1.0,
        "bias": 0.0,
        "source": "derived",
        "description": "ret_bar (bar-to-bar return), vol_proxy (volatility proxy from ATR)"
    })

    # Block 6: Agent state features (indices 23-28)
    layout.append({
        "name": "agent",
        "size": 6,
        "dtype": "float32",
        "clip": DEFAULT_TANH_CLIP,
        "scale": 1.0,
        "bias": 0.0,
        "source": "agent",
        "description": "cash_ratio, position_ratio, vol_imbalance, trade_intensity, realized_spread, fill_ratio"
    })

    # Block 7: Microstructure proxies (indices 29-31)
    layout.append({
        "name": "microstructure",
        "size": 3,
        "dtype": "float32",
        "clip": DEFAULT_TANH_CLIP,
        "scale": 1.0,
        "bias": 0.0,
        "source": "micro",
        "description": "price_momentum, bb_squeeze, trend_strength (MACD divergence)"
    })

    # Block 8: Bollinger Bands context (indices 32-33)
    # NOTE: This was missing from previous versions!
    layout.append({
        "name": "bb_context",
        "size": 2,
        "dtype": "float32",
        "clip": None,
        "scale": 1.0,
        "bias": 0.0,
        "source": "indicators",
        "description": "bb_position (price within bands), bb_width_norm (band width normalized)"
    })

    # Block 9: Event metadata (indices 34-38)
    layout.append({
        "name": "metadata",
        "size": 5,
        "dtype": "float32",
        "clip": DEFAULT_TANH_CLIP,
        "scale": 1.0,
        "bias": 0.0,
        "source": "meta",
        "description": "is_high_importance, time_since_event, risk_off_flag, fear_greed_value, fear_greed_indicator"
    })
    # External normalized columns block (if any)
    if ext_dim and ext_dim > 0:
        layout.append({
            "name": "external",
            "size": ext_dim,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "external"
        })
    # Token metadata block (num_tokens_norm, token_id_norm)
    if max_tokens > 0:
        layout.append({
            "name": "token_meta",
            "size": 2,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "token_meta"
        })
    # Token one-hot block
    if max_tokens > 0:
        layout.append({
            "name": "token",
            "size": max_tokens,
            "dtype": "float32",
            "clip": None,
            "scale": 1.0,
            "bias": 0.0,
            "source": "token"
        })
    FEATURES_LAYOUT = layout
    # Compute total feature vector length
    N_FEATURES = sum(block["size"] for block in layout)
    return FEATURES_LAYOUT

# Build default layout with default parameters on import
make_layout({})
