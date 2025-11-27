# -*- coding: utf-8 -*-
"""
data_loader_multi_asset.py
------------------------------------------------------------------
Unified data loader for multi-asset support (crypto + stocks).

This module extends fetch_all_data_patch.py with:
- Support for stock data from Alpaca and Polygon
- Trading hours awareness for equities
- Asset-class specific preprocessing
- Unified data format for TradingEnv

Usage:
    from data_loader_multi_asset import load_multi_asset_data, AssetClass

    # Load crypto data
    crypto_dfs, crypto_obs = load_multi_asset_data(
        paths=["data/processed/BTCUSDT.feather"],
        asset_class=AssetClass.CRYPTO,
    )

    # Load stock data
    stock_dfs, stock_obs = load_multi_asset_data(
        paths=["data/stocks/AAPL.parquet"],
        asset_class=AssetClass.EQUITY,
    )

    # Or via adapter
    stock_dfs, stock_obs = load_from_adapter(
        vendor="alpaca",
        symbols=["AAPL", "MSFT", "GOOGL"],
        timeframe="4h",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class AssetClass(str, Enum):
    """Asset class for data loading."""
    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"


class DataVendor(str, Enum):
    """Data source vendor."""
    BINANCE = "binance"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    CSV = "csv"
    FEATHER = "feather"
    PARQUET = "parquet"


# =============================================================================
# TIMEFRAME UTILITIES
# =============================================================================

def timeframe_to_seconds(tf: str) -> int:
    """Convert timeframe string to seconds."""
    tf_lower = tf.lower().strip()

    mappings = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
        "1w": 604800,
    }

    if tf_lower in mappings:
        return mappings[tf_lower]

    # Parse numeric format (e.g., "15min", "4hour")
    import re
    match = re.match(r"(\d+)(m|min|h|hour|d|day|w|week)", tf_lower)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        unit_map = {
            "m": 60, "min": 60,
            "h": 3600, "hour": 3600,
            "d": 86400, "day": 86400,
            "w": 604800, "week": 604800,
        }
        return value * unit_map.get(unit, 60)

    # Default to 4h
    logger.warning(f"Unknown timeframe '{tf}', defaulting to 4h")
    return 14400


def align_timestamp(ts: int, timeframe_seconds: int) -> int:
    """Align timestamp to timeframe boundary (floor)."""
    return (ts // timeframe_seconds) * timeframe_seconds


# =============================================================================
# FEAR & GREED LOADER (CRYPTO)
# =============================================================================

FNG_PATH = os.path.join("data", "fear_greed.csv")


def load_fear_greed(
    timeframe_seconds: int = 14400,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load Fear & Greed index data for crypto.

    Args:
        timeframe_seconds: Timeframe for alignment
        path: Path to fear_greed.csv

    Returns:
        DataFrame with timestamp, fear_greed_value, fear_greed_value_norm
    """
    fng_path = path or FNG_PATH

    if not os.path.exists(fng_path):
        return pd.DataFrame(columns=["timestamp", "fear_greed_value"])

    fng = pd.read_csv(fng_path)

    # Find timestamp and value columns
    cols = {c.lower(): c for c in fng.columns}
    ts_col = "timestamp" if "timestamp" in cols else next(
        (c for c in fng.columns if "time" in c.lower()), "timestamp"
    )
    val_col = "fear_greed_value" if "fear_greed_value" in fng.columns else (
        "value" if "value" in fng.columns else None
    )

    fng = fng.rename(columns={ts_col: "timestamp"})
    if val_col and val_col != "fear_greed_value":
        fng = fng.rename(columns={val_col: "fear_greed_value"})

    # Normalize timestamp to seconds
    if fng["timestamp"].max() > 10_000_000_000:
        fng["timestamp"] = (fng["timestamp"] // 1000).astype("int64")
    else:
        fng["timestamp"] = fng["timestamp"].astype("int64")

    fng = fng[["timestamp", "fear_greed_value"]].copy()
    fng["fear_greed_value_norm"] = fng["fear_greed_value"].astype(float) / 100.0

    # Align to timeframe
    fng["timestamp"] = fng["timestamp"].apply(
        lambda x: align_timestamp(x, timeframe_seconds)
    )
    fng = fng.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    return fng


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_from_file(
    path: Union[str, Path],
    asset_class: AssetClass = AssetClass.CRYPTO,
    timeframe: str = "4h",
) -> pd.DataFrame:
    """
    Load OHLCV data from file (CSV, Feather, Parquet).

    Args:
        path: Path to data file
        asset_class: Asset class (crypto/equity)
        timeframe: Timeframe for alignment

    Returns:
        DataFrame with standardized columns
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Extract symbol from filename if not in data
    symbol = path.stem.upper()
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    # Standardize columns
    df = _standardize_columns(df, asset_class, timeframe)

    return df


def _standardize_columns(
    df: pd.DataFrame,
    asset_class: AssetClass,
    timeframe: str = "4h",
) -> pd.DataFrame:
    """
    Standardize DataFrame columns for TradingEnv.

    Args:
        df: Raw DataFrame
        asset_class: Asset class
        timeframe: Timeframe for alignment

    Returns:
        Standardized DataFrame
    """
    df = df.copy()
    timeframe_seconds = timeframe_to_seconds(timeframe)

    # Column name mapping for different sources
    column_aliases = {
        "timestamp": ["timestamp", "ts", "time", "date", "datetime", "close_time", "t"],
        "open": ["open", "o", "Open"],
        "high": ["high", "h", "High"],
        "low": ["low", "l", "Low"],
        "close": ["close", "c", "Close"],
        "volume": ["volume", "v", "Volume", "vol"],
    }

    # Rename columns using aliases
    for target, aliases in column_aliases.items():
        if target not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: target})
                    break

    # Handle timestamp
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            ts = pd.to_numeric(df["open_time"], errors="coerce")
            if ts.max() > 10_000_000_000:
                ts = ts // 1000
            df["timestamp"] = ts + timeframe_seconds
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**9
        else:
            raise ValueError("No timestamp column found")
    else:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.max() > 10_000_000_000:
            ts = ts // 1000
        df["timestamp"] = ts.astype("int64")

    # Align timestamp to timeframe
    df["timestamp"] = df["timestamp"].apply(
        lambda x: align_timestamp(int(x), timeframe_seconds)
    )

    # Ensure OHLCV columns exist
    required_float = ["open", "high", "low", "close", "volume"]
    for col in required_float:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].astype(float)

    # Add optional columns with defaults
    optional_columns = {
        "quote_asset_volume": lambda: df["close"] * df["volume"],
        "number_of_trades": lambda: 0,
        "taker_buy_base_asset_volume": lambda: 0.0,
        "taker_buy_quote_asset_volume": lambda: 0.0,
        "vwap": lambda: (df["high"] + df["low"] + df["close"]) / 3,
    }

    for col, default_fn in optional_columns.items():
        if col not in df.columns:
            df[col] = default_fn()

    # Asset-class specific handling
    if asset_class == AssetClass.EQUITY:
        # Add trading hours flag
        if "is_trading_hours" not in df.columns:
            df["is_trading_hours"] = True  # Assume data is already filtered

        # No Fear & Greed for stocks (use VIX or similar)
        if "fear_greed_value" not in df.columns:
            df["fear_greed_value"] = np.nan

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    # Standardize column order
    base_cols = [
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + other_cols]

    return df


def load_from_adapter(
    vendor: Union[str, DataVendor],
    symbols: Sequence[str],
    timeframe: str = "4h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load data from market data adapter.

    Args:
        vendor: Data vendor (alpaca, polygon, binance)
        symbols: List of symbols to load
        timeframe: Bar timeframe
        start_date: Start date (ISO format or YYYY-MM-DD)
        end_date: End date
        config: Adapter configuration

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    if isinstance(vendor, str):
        vendor = DataVendor(vendor.lower())

    config = config or {}

    # Determine asset class from vendor
    asset_class = AssetClass.CRYPTO
    if vendor in (DataVendor.ALPACA, DataVendor.POLYGON):
        asset_class = AssetClass.EQUITY

    # Parse dates
    start_ts = None
    end_ts = None
    if start_date:
        start_dt = pd.Timestamp(start_date, tz="UTC")
        start_ts = int(start_dt.timestamp() * 1000)
    if end_date:
        end_dt = pd.Timestamp(end_date, tz="UTC")
        end_ts = int(end_dt.timestamp() * 1000)

    # Create adapter and fetch data
    all_dfs: Dict[str, pd.DataFrame] = {}
    all_obs: Dict[str, np.ndarray] = {}

    try:
        from adapters.registry import create_market_data_adapter

        adapter = create_market_data_adapter(vendor.value, config)

        for symbol in symbols:
            logger.info(f"Loading {symbol} from {vendor.value}")

            bars = adapter.get_bars(
                symbol,
                timeframe,
                limit=50000,
                start_ts=start_ts,
                end_ts=end_ts,
            )

            if not bars:
                logger.warning(f"No data returned for {symbol}")
                continue

            # Convert bars to DataFrame
            rows = []
            for bar in bars:
                rows.append({
                    "timestamp": bar.ts // 1000 if bar.ts > 10_000_000_000 else bar.ts,
                    "symbol": symbol.upper(),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume_base),
                    "quote_asset_volume": float(bar.volume_quote or 0),
                    "number_of_trades": bar.trades or 0,
                    "taker_buy_base_asset_volume": 0.0,
                    "taker_buy_quote_asset_volume": 0.0,
                    "vwap": float(bar.vwap) if bar.vwap else None,
                })

            df = pd.DataFrame(rows)
            df = _standardize_columns(df, asset_class, timeframe)
            all_dfs[symbol.upper()] = df

    except ImportError as e:
        logger.error(f"Adapter not available: {e}")
        raise

    return all_dfs, all_obs


def load_multi_asset_data(
    paths: Sequence[Union[str, Path]],
    asset_class: AssetClass = AssetClass.CRYPTO,
    timeframe: str = "4h",
    merge_fear_greed: bool = True,
    synthetic_fraction: float = 0.0,
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load data from multiple files with multi-asset support.

    This is the main entry point for loading training data,
    compatible with fetch_all_data_patch.load_all_data().

    Args:
        paths: List of file paths (feather, parquet, csv)
        asset_class: Asset class (crypto/equity)
        timeframe: Bar timeframe
        merge_fear_greed: Merge Fear & Greed data (crypto only)
        synthetic_fraction: Fraction of synthetic data (unused)
        seed: Random seed (unused)

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    all_dfs: Dict[str, pd.DataFrame] = {}
    all_obs: Dict[str, np.ndarray] = {}

    timeframe_seconds = timeframe_to_seconds(timeframe)

    # Load Fear & Greed for crypto
    fng = pd.DataFrame()
    if asset_class == AssetClass.CRYPTO and merge_fear_greed:
        fng = load_fear_greed(timeframe_seconds)

    for path in paths:
        try:
            df = load_from_file(path, asset_class, timeframe)
            symbol = df["symbol"].iloc[0] if "symbol" in df.columns else Path(path).stem

            # Merge Fear & Greed for crypto
            if not fng.empty and asset_class == AssetClass.CRYPTO:
                df = _merge_fear_greed(df, fng)

            all_dfs[symbol] = df
            logger.debug(f"Loaded {symbol}: {len(df)} rows")

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No data files were successfully loaded")

    logger.info(f"Loaded {len(all_dfs)} symbols: {list(all_dfs.keys())}")
    return all_dfs, all_obs


def _merge_fear_greed(df: pd.DataFrame, fng: pd.DataFrame) -> pd.DataFrame:
    """Merge Fear & Greed data into DataFrame."""
    if fng.empty:
        return df

    # Preserve original if exists
    orig_col = None
    if "fear_greed_value" in df.columns:
        orig_col = "fear_greed_value_orig"
        df = df.rename(columns={"fear_greed_value": orig_col})

    # Merge with backward fill
    fng_sorted = fng.sort_values("timestamp")[["timestamp", "fear_greed_value"]].copy()
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        fng_sorted,
        on="timestamp",
        direction="backward"
    )

    if "fear_greed_value" in df.columns:
        df["fear_greed_value"] = df["fear_greed_value"].ffill()

    # Restore original if merge failed
    if "fear_greed_value" not in df.columns and orig_col:
        df["fear_greed_value"] = df[orig_col]
    elif "fear_greed_value" in df.columns and orig_col:
        df["fear_greed_value"] = df["fear_greed_value"].fillna(df[orig_col])

    if orig_col and orig_col in df.columns:
        df = df.drop(columns=[orig_col])

    return df


# =============================================================================
# TRADING HOURS FILTERING (EQUITY)
# =============================================================================

def filter_trading_hours(
    df: pd.DataFrame,
    include_extended: bool = False,
    timezone_str: str = "America/New_York",
) -> pd.DataFrame:
    """
    Filter DataFrame to trading hours only (for equities).

    Args:
        df: DataFrame with timestamp column
        include_extended: Include pre-market (4:00-9:30) and after-hours (16:00-20:00)
        timezone_str: Market timezone

    Returns:
        Filtered DataFrame
    """
    try:
        import pytz
        tz = pytz.timezone(timezone_str)
    except ImportError:
        logger.warning("pytz not installed - cannot filter trading hours")
        return df

    df = df.copy()

    # Convert timestamp to datetime
    df["_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["_dt_local"] = df["_dt"].dt.tz_convert(tz)

    # Extract time components
    df["_hour"] = df["_dt_local"].dt.hour
    df["_minute"] = df["_dt_local"].dt.minute
    df["_weekday"] = df["_dt_local"].dt.weekday  # 0=Monday, 6=Sunday

    # Filter weekdays only
    mask = df["_weekday"] < 5  # Monday-Friday

    # Time of day filter
    time_minutes = df["_hour"] * 60 + df["_minute"]

    if include_extended:
        # 4:00 - 20:00 ET
        mask &= (time_minutes >= 240) & (time_minutes < 1200)
    else:
        # 9:30 - 16:00 ET
        mask &= (time_minutes >= 570) & (time_minutes < 960)

    # Clean up temp columns
    df = df.drop(columns=["_dt", "_dt_local", "_hour", "_minute", "_weekday"])

    return df[mask].copy()


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(
    df: pd.DataFrame,
    asset_class: AssetClass = AssetClass.CRYPTO,
    min_rows: int = 100,
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame for training.

    Args:
        df: DataFrame to validate
        asset_class: Asset class
        min_rows: Minimum required rows

    Returns:
        (is_valid, list of error messages)
    """
    errors: List[str] = []

    # Check row count
    if len(df) < min_rows:
        errors.append(f"Insufficient data: {len(df)} rows (min: {min_rows})")

    # Check required columns
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check for NaN in critical columns
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                errors.append(f"Column '{col}' has {nan_count} NaN values")

    # Check OHLC validity
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        invalid_count = invalid_ohlc.sum()
        if invalid_count > 0:
            errors.append(f"{invalid_count} rows have invalid OHLC (high < low, etc.)")

    # Check timestamp ordering
    if "timestamp" in df.columns:
        if not df["timestamp"].is_monotonic_increasing:
            errors.append("Timestamps are not monotonically increasing")

    # Check for gaps in time series
    if "timestamp" in df.columns and len(df) > 1:
        diffs = df["timestamp"].diff().dropna()
        expected_diff = diffs.mode().iloc[0] if len(diffs) > 0 else None
        if expected_diff:
            gaps = (diffs > expected_diff * 2).sum()
            if gaps > len(df) * 0.1:  # More than 10% gaps
                errors.append(f"Time series has {gaps} gaps (>10% of data)")

    is_valid = len(errors) == 0
    return is_valid, errors


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_crypto_data(
    paths: Sequence[Union[str, Path]],
    timeframe: str = "4h",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """Load crypto data from files."""
    return load_multi_asset_data(
        paths=paths,
        asset_class=AssetClass.CRYPTO,
        timeframe=timeframe,
        merge_fear_greed=True,
    )


def load_stock_data(
    paths: Sequence[Union[str, Path]],
    timeframe: str = "4h",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """Load stock data from files."""
    return load_multi_asset_data(
        paths=paths,
        asset_class=AssetClass.EQUITY,
        timeframe=timeframe,
        merge_fear_greed=False,
    )


def load_alpaca_data(
    symbols: Sequence[str],
    timeframe: str = "4h",
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load stock data from Alpaca API.

    Args:
        symbols: List of stock symbols
        timeframe: Bar timeframe
        start_date: Start date
        end_date: End date (default: today)
        api_key: Alpaca API key (or env ALPACA_API_KEY)
        api_secret: Alpaca API secret

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    config = {}
    if api_key:
        config["api_key"] = api_key
    if api_secret:
        config["api_secret"] = api_secret

    return load_from_adapter(
        vendor=DataVendor.ALPACA,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )


def load_polygon_data(
    symbols: Sequence[str],
    timeframe: str = "4h",
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Load stock data from Polygon.io API.

    Args:
        symbols: List of stock symbols
        timeframe: Bar timeframe
        start_date: Start date
        end_date: End date (default: today)
        api_key: Polygon API key (or env POLYGON_API_KEY)

    Returns:
        (all_dfs_dict, all_obs_dict) tuple
    """
    config = {}
    if api_key:
        config["api_key"] = api_key

    return load_from_adapter(
        vendor=DataVendor.POLYGON,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
